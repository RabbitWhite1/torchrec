#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
from math import factorial
import os
import sys
import os.path as osp
from typing import cast, Iterator, List
from datetime import datetime

import torch
import torchmetrics as metrics
from pyre_extensions import none_throws
from torch import nn, distributed as dist
from torch.utils.data import DataLoader
from torchrec import EmbeddingBagCollection
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.datasets.utils import Batch
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner.types import Topology
from torchrec.distributed.types import ModuleSharder
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from tqdm import tqdm
import rich
from rich.progress import track, Progress
import rich.progress as rich_progress
from rich.console import Console
from rich.traceback import install
from torchrec.test_utils.timer import Timer
from torchrec.distributed.model_parallel import get_default_sharders

install(show_locals=False)
console = Console()

from myplanner import MyShardingPlanner


# OSS import
try:
    # pyre-ignore[21]
    # @manual=//torchrec/github/examples/dlrm/data:dlrm_dataloader
    from data.dlrm_dataloader import get_dataloader, STAGES

    # pyre-ignore[21]
    # @manual=//torchrec/github/examples/dlrm/modules:dlrm_train
    from modules.dlrm_train import DLRMTrain
except ImportError:
    pass

# internal import
try:
    from .data.dlrm_dataloader import (  # noqa F811
        get_dataloader,
        STAGES,
    )
    from .modules.dlrm_train import DLRMTrain  # noqa F811
except ImportError:
    pass

TRAIN_PIPELINE_STAGES = 3  # Number of stages in TrainPipelineSparseDist.


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec dlrm example trainer")
    ################################ env
    parser.add_argument(
        "--env_first", action='store_true',
        help="If used, then use args in environmental variables first."
    )
    ################################ progress
    parser.add_argument(
        "--only_test", action='store_true',
        help="Only do testing."
    )
    ################################ epochs/batches
    parser.add_argument(
        "--epochs", type=int, default=1, help="number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size to use for training"
    )
    parser.add_argument(
        "--limit_train_batches",
        type=int,
        default=1,
        help="number of train batches",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=int,
        default=None,
        help="number of validation batches",
    )
    parser.add_argument(
        "--limit_test_batches",
        type=int,
        default=None,
        help="number of test batches",
    )
    parser.add_argument(
        "--shuffle_batches",
        type=bool,
        default=False,
        help="Shuffle each batch during training.",
    )
    ################################ dist/parallel
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="number of dataloader workers",
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dist_backend", type=str, default="")
    ################################ embedding tables
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=100_000,
        help="max_ind_size. The number of embeddings in each embedding table. Defaults"
        " to 100_000 if num_embeddings_per_feature is not supplied.",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default=None,
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Size of each embedding.",
    )
    parser.add_argument(
        "--embedding_dim_per_feature",
        type=str,
        default=None,
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--num_features_per_rank",
        type=str,
        default=None,
        help="Number of features per rank. If specified, the auto-sharding will be disabled",
    )
    parser.add_argument(
        "--pooling_factor_per_feature",
        type=str,
        default=None,
        help="Pooling factors per feature. ",
    )
    parser.add_argument(
        "--autoplan",
        type=bool,
        default=True,
        help="Auto plan or use `pooling_factor_per_feature`",
    )
    ################################ dense layer
    parser.add_argument(
        "--dense_arch_layer_sizes",
        type=str,
        default="512,256,64",
        help="Comma separated layer sizes for dense arch.",
    )
    ################################ over layer
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=str,
        default="512,512,256,1",
        help="Comma separated layer sizes for over arch.",
    )
    ################################ datasets
    parser.add_argument(
        "--dataset_type",
        type=str,
        default='random',
        help="dataset type. random|realworld|synthetic"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="criteo_1t",
        help="dataset for experiment, current support criteo_1tb, criteo_kaggle",
    )
    parser.add_argument(
        "--in_memory_binary_criteo_path",
        type=str,
        default=None,
        help="Path to a folder containing the binary (npy) files for the Criteo dataset."
        " When supplied, InMemoryBinaryCriteoIterDataPipe is used.",
    )
    ################################ log
    parser.add_argument(
        "--conf_id",
        type=int,
        help="Execution Configuration Id",
    )
    parser.add_argument(
        "--logs_dirname",
        type=str,
        help="Logs dirname",
    )
    parser.add_argument(
        "--plans_dirname",
        type=str,
        help="Plans dirname",
    )
    ################################ other args
    parser.add_argument(
        "--undersampling_rate",
        type=float,
        help="Desired proportion of zero-labeled samples to retain (i.e. undersampling zero-labeled rows)."
        " Ex. 0.3 indicates only 30pct of the rows with label 0 will be kept."
        " All rows with label 1 will be kept. Value should be between 0 and 1."
        " When not supplied, no undersampling occurs.",
    )
    parser.add_argument(
        "--seed",
        type=float,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--pin_memory",
        dest="pin_memory",
        action="store_true",
        help="Use pinned memory when loading data.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate.",
    )
    
    parser.set_defaults(pin_memory=None)
    args = parser.parse_args(argv)
    from_env_args = [
        'batch_size',
        'dataset_type',
        'embedding_dim_per_feature',
        'pooling_factor_per_feature',
        'num_embeddings_per_feature',
        'num_features_per_rank',
        'autoplan',
    ]
    for from_env_arg in from_env_args:
        exec(f'args.{from_env_arg} = (os.getenv("{from_env_arg}", args.{from_env_arg}))')
    args.batch_size = int(args.batch_size)
    if args.autoplan in [1, 'True', '1']:
        args.autoplan = True
    elif args.autoplan in [0, 'False', '0']:
        args.autoplan = False
    else:
        raise RuntimeError('error parsing env autoplan')

    return args


def _evaluate(
    args: argparse.Namespace,
    train_pipeline: TrainPipelineSparseDist,
    iterator: Iterator[Batch],
    next_iterator: Iterator[Batch],
    stage: str,
) -> None:
    """
    Evaluate model. Computes and prints metrics including AUROC and Accuracy. Helper
    function for train_val_test.

    Args:
        args (argparse.Namespace): parsed command line args.
        train_pipeline (TrainPipelineSparseDist): pipelined model.
        iterator (Iterator[Batch]): Iterator used for val/test batches.
        next_iterator (Iterator[Batch]): Iterator used for the next phase (either train
            if there are more epochs to train on or test if all epochs are complete).
            Used to queue up the next TRAIN_PIPELINE_STAGES - 1 batches before
            train_val_test switches to the next phase. This is done so that when the
            next phase starts, the first output train_pipeline generates an output for
            is the 1st batch for that phase.
        stage (str): "val" or "test".

    Returns:
        None.
    """
    model = train_pipeline._model
    model.eval()
    device = train_pipeline._device
    limit_batches = (
        args.limit_val_batches if stage == "val" else args.limit_test_batches
    )
    if limit_batches is not None:
        limit_batches -= TRAIN_PIPELINE_STAGES - 1

    # Because TrainPipelineSparseDist buffer batches internally, we load in
    # TRAIN_PIPELINE_STAGES - 1 batches from the next_iterator into the buffers so that
    # when train_val_test switches to the next phase, train_pipeline will start
    # producing results for the TRAIN_PIPELINE_STAGES - 1 buffered batches (as opposed
    # to the last TRAIN_PIPELINE_STAGES - 1 batches from iterator).
    combined_iterator = itertools.chain(
        iterator
        if limit_batches is None
        else itertools.islice(iterator, limit_batches),
        itertools.islice(next_iterator, TRAIN_PIPELINE_STAGES - 1),
    )
    # auroc = metrics.AUROC(compute_on_step=False).to(device)
    accuracy = metrics.Accuracy(compute_on_step=False).to(device)

    # Infinite iterator instead of while-loop to leverage tqdm progress bar.
    for _ in tqdm(range(getattr(args, f'limit_{stage}_batches')), desc=f"Evaluating {stage} set"):
        try:
            _loss, logits, labels = train_pipeline.progress(combined_iterator)
            # auroc(logits, labels)
            accuracy(logits, labels)
        except StopIteration:
            break
    # auroc_result = auroc.compute().item()
    accuracy_result = accuracy.compute().item()
    if dist.get_rank() == 0:
        # print(f"AUROC over {stage} set: {auroc_result}.")
        print(f"Accuracy over {stage} set: {accuracy_result}.")


def _train(
    args: argparse.Namespace,
    train_pipeline: TrainPipelineSparseDist,
    iterator: Iterator[Batch],
    next_iterator: Iterator[Batch],
    epoch: int,
) -> None:
    """
    Train model for 1 epoch. Helper function for train_val_test.

    Args:
        args (argparse.Namespace): parsed command line args.
        train_pipeline (TrainPipelineSparseDist): pipelined model.
        iterator (Iterator[Batch]): Iterator used for training batches.
        next_iterator (Iterator[Batch]): Iterator used for validation batches. Used to
            queue up the next TRAIN_PIPELINE_STAGES - 1 batches before train_val_test
            switches to validation mode. This is done so that when validation starts,
            the first output train_pipeline generates an output for is the 1st
            validation batch (as opposed to a buffered train batch).
        epoch (int): Which epoch the model is being trained on.

    Returns:
        None.
    """
    train_pipeline._model.train()

    limit_batches = args.limit_train_batches
    # For the first epoch, train_pipeline has no buffered batches, but for all other
    # epochs, train_pipeline will have TRAIN_PIPELINE_STAGES - 1 from iterator already
    # present in its buffer.
    if limit_batches is not None and epoch > 0:
        limit_batches -= TRAIN_PIPELINE_STAGES - 1

    # Because TrainPipelineSparseDist buffer batches internally, we load in
    # TRAIN_PIPELINE_STAGES - 1 batches from the next_iterator into the buffers so that
    # when train_val_test switches to the next phase, train_pipeline will start
    # producing results for the TRAIN_PIPELINE_STAGES - 1 buffered batches (as opposed
    # to the last TRAIN_PIPELINE_STAGES - 1 batches from iterator).
    combined_iterator = itertools.chain(
        iterator
        if args.limit_train_batches is None
        else itertools.islice(iterator, limit_batches),
        itertools.islice(next_iterator, TRAIN_PIPELINE_STAGES - 1),
    )

    # Infinite iterator instead of while-loop to leverage tqdm progress bar.
    for _ in tqdm(range(args.limit_train_batches), desc=f"Epoch {epoch}"):
        try:
            train_pipeline.progress(combined_iterator)
        except StopIteration:
            break


def train_val_test(
    args: argparse.Namespace,
    train_pipeline: TrainPipelineSparseDist,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
) -> None:
    """
    Train/validation/test loop. Contains customized logic to ensure each dataloader's
    batches are used for the correct designated purpose (train, val, test). This logic
    is necessary because TrainPipelineSparseDist buffers batches internally (so we
    avoid batches designated for one purpose like training getting buffered and used for
    another purpose like validation).

    Args:
        args (argparse.Namespace): parsed command line args.
        train_pipeline (TrainPipelineSparseDist): pipelined model.
        train_dataloader (DataLoader): DataLoader used for training.
        val_dataloader (DataLoader): DataLoader used for validation.
        test_dataloader (DataLoader): DataLoader used for testing.

    Returns:
        None.
    """
    
    rank = dist.get_rank()
    dist.barrier()
    Timer.set_base_time(rank, datetime.now().timestamp())
    test_iterator = iter(test_dataloader)
    if not args.only_test:
        train_iterator = iter(train_dataloader)
        for epoch in range(args.epochs):
            val_iterator = iter(val_dataloader)
            _train(args, train_pipeline, train_iterator, val_iterator, epoch)
            train_iterator = iter(train_dataloader)
            val_next_iterator = (
                test_iterator if epoch == args.epochs - 1 else train_iterator
            )
            _evaluate(args, train_pipeline, val_iterator, val_next_iterator, "val")

    _evaluate(args, train_pipeline, test_iterator, iter(test_dataloader), "test")


def main(argv: List[str]) -> None:
    """
    Trains, validates, and tests a Deep Learning Recommendation Model (DLRM)
    (https://arxiv.org/abs/1906.00091). The DLRM model contains both data parallel
    components (e.g. multi-layer perceptrons & interaction arch) and model parallel
    components (e.g. embedding tables). The DLRM model is pipelined so that dataloading,
    data-parallel to model-parallel comms, and forward/backward are overlapped. Can be
    run with either a random dataloader or an in-memory Criteo 1 TB click logs dataset
    (https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/).

    Args:
        argv (List[str]): command line args.

    Returns:
        None.
    """
    args = parse_args(argv)

    rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        device: torch.device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device: torch.device = torch.device("cpu")
        backend = "gloo"

    if not torch.distributed.is_initialized():
        dist.init_process_group(backend=backend)
        Timer.init(dist.get_world_size())

    # Sets default limits for random dataloader iterations when left unspecified.
    if args.in_memory_binary_criteo_path is None:
        # pyre-ignore[16]
        for stage in STAGES:
            attr = f"limit_{stage}_batches"
            if getattr(args, attr) is None:
                setattr(args, attr, 10)
    # Process embeddings args
    if args.num_embeddings_per_feature is not None:
        args.num_embeddings_per_feature = list(map(int, args.num_embeddings_per_feature.split(",")))
        args.num_embeddings = None
        setattr(args, 'sparse_feature_names', [f"cat_{idx}" for idx in range(len(args.num_embeddings_per_feature))])
        setattr(args, 'num_features', [f"cat_{idx}" for idx in range(len(args.num_embeddings_per_feature))])
    else:
        raise RuntimeError("unhandled")
    if args.embedding_dim_per_feature is not None:
        args.embedding_dim_per_feature = list(map(int, args.embedding_dim_per_feature.split(',')))
        args.embedding_dim = None
    if args.pooling_factor_per_feature is not None:
        args.pooling_factor_per_feature = list(map(int, args.pooling_factor_per_feature.split(',')))
    if args.num_features_per_rank is not None:
        args.num_features_per_rank = list(map(int, args.num_features_per_rank.split(',')))

    # args sanity check
    assert len(args.num_embeddings_per_feature) == len(args.embedding_dim_per_feature)
    assert len(args.embedding_dim_per_feature) == len(args.pooling_factor_per_feature)
    assert len(args.pooling_factor_per_feature) == sum(args.num_features_per_rank)

    rich.print(args)
    dist.barrier()
    rich.print(f'{backend=}; {os.environ["LOCAL_RANK"]=}; {os.environ["RANK"]=}; {dist.get_rank()=}; '
               f'{os.environ["WORLD_SIZE"]=}; {os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}')
    
    # TODO add CriteoIterDataPipe support and add random_dataloader arg
    # pyre-ignore[16]
    train_dataloader = get_dataloader(args, backend, "train")
    # pyre-ignore[16]
    val_dataloader = get_dataloader(args, backend, "val")
    # pyre-ignore[16]
    test_dataloader = get_dataloader(args, backend, "test")

    eb_configs = [
        EmbeddingBagConfig(
            name=f"{feature_name}",
            embedding_dim=none_throws(args.embedding_dim_per_feature)[feature_idx],
            num_embeddings=none_throws(args.num_embeddings_per_feature)[feature_idx],
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(args.sparse_feature_names)
    ]
    sharded_module_kwargs = {}
    if args.over_arch_layer_sizes is not None:
        sharded_module_kwargs["over_arch_layer_sizes"] = list(
            map(int, args.over_arch_layer_sizes.split(","))
        )

    train_model = DLRMTrain(
        embedding_bag_collection=EmbeddingBagCollection(
            tables=eb_configs, device=torch.device("meta")
        ),
        dense_in_features=len(DEFAULT_INT_NAMES),
        dense_arch_layer_sizes=list(map(int, args.dense_arch_layer_sizes.split(","))),
        over_arch_layer_sizes=list(map(int, args.over_arch_layer_sizes.split(","))),
        dense_device=device,
        interaction_type='cat'
    )
    fused_params = {
        "learning_rate": args.learning_rate,
    }
    sharders = [
        EmbeddingBagCollectionSharder(fused_params=fused_params),
    ]
    topology = Topology(world_size=dist.get_world_size(), compute_device='cpu')

    feature_name_to_rank = {}
    for feature_idx, feature_name in enumerate(args.sparse_feature_names):
        for r, num_features in enumerate(args.num_features_per_rank):
            if feature_idx >= num_features:
                feature_idx -= num_features
            else:
                feature_name_to_rank[feature_name] = r
                break
    feature_name_to_pooling_factor = {
        feature_name: pooling_factor
        for feature_name, pooling_factor 
        in zip(args.sparse_feature_names, args.pooling_factor_per_feature)
    }
    if args.autoplan:
        sharding_plan = None
    else:
        my_sharding_planner = MyShardingPlanner(feature_name_to_rank, feature_name_to_pooling_factor, topology)
        sharding_plan = my_sharding_planner.plan(
            train_model,
            sharders=get_default_sharders()
        )

    begin_dmp_time = datetime.now()
    model = DistributedModelParallel(
        module=train_model,
        device=device,
        sharders=cast(List[ModuleSharder[nn.Module]], sharders),
        plan=sharding_plan,
    )
    end_dmp_time = datetime.now()
    rich.print(f'{model=}')
    rich.print(f'dmp time: {end_dmp_time - begin_dmp_time}')
    dense_optimizer = KeyedOptimizerWrapper(
        dict(model.named_parameters()),
        lambda params: torch.optim.SGD(params, lr=args.learning_rate),
    )
    optimizer = CombinedOptimizer([model.fused_optimizer, dense_optimizer])

    train_pipeline = TrainPipelineSparseDist(
        model,
        optimizer,
        device,
    )
    train_val_test(
        args, train_pipeline, train_dataloader, val_dataloader, test_dataloader
    )
    os.makedirs(args.logs_dirname, exist_ok=True)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    log_path = osp.join(args.logs_dirname, f'log{args.conf_id}--RANK{world_size}_{rank}.csv')
    Timer.stats(rank=dist.get_rank()).to_csv(log_path)
    print(f'stats outputs to {log_path}')
    os.makedirs(args.plans_dirname, exist_ok=True)
    plan_path = osp.join(args.plans_dirname, f'plan{args.conf_id}--RANK{world_size}_{rank}.log')
    with open(plan_path, 'w') as f:
        f.write(str(model.plan))
        print(f'plan outputs to {plan_path}')


if __name__ == "__main__":
    main(sys.argv[1:])
