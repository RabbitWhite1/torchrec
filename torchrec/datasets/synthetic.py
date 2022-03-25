#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterator, List, Optional

import torch
from pyre_extensions import none_throws
from torch.utils.data.dataset import IterableDataset
from torchrec.datasets.utils import Batch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class _SyntheticRecBatch:
    generator: Optional[torch.Generator]

    def __init__(
        self,
        keys: List[str],
        batch_size: int,
        pooling_factor_per_feature: List[int],
        num_embeddings_per_feature: List[int],
        num_dense: int,
        manual_seed: Optional[int] = None,
    ) -> None:
        self.keys = keys
        self.num_keys: int = len(keys)
        self.batch_size = batch_size
        self.pooling_factor_per_feature = pooling_factor_per_feature
        self.num_embeddings_per_feature = num_embeddings_per_feature
        self.num_dense = num_dense

        if manual_seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(manual_seed)
        else:
            self.generator = None

        self.iter_num = 0
        self._pooling_factor_in_batch: int = (
            sum(self.pooling_factor_per_feature) * self.batch_size
        )
        # `Pooling factor` * B * F
        self.max_values: torch.Tensor = torch.tensor(
            [
                num_embeddings
                for pooling_factor, num_embeddings in zip(pooling_factor_per_feature, num_embeddings_per_feature)
                for pf in range(pooling_factor)
                for b in range(batch_size)
            ]
        )
        self._generated_batches: List[Batch] = [self._generate_batch()] * 10
        self.batch_index = 0

    def __iter__(self) -> "_SyntheticRecBatch":
        return self

    def __next__(self) -> Batch:
        batch = self._generated_batches[self.batch_index % len(self._generated_batches)]
        self.batch_index += 1
        return batch

    def _generate_batch(self) -> Batch:
        values = (
            torch.rand(
                self._pooling_factor_in_batch,
                generator=self.generator,
            )
            * none_throws(self.max_values)
        ).type(torch.LongTensor)
        sparse_features = KeyedJaggedTensor.from_lengths_sync(
            keys=self.keys,
            values=values,
            lengths=torch.tensor(
                [
                    pooling_factor
                    for pooling_factor in self.pooling_factor_per_feature
                    for b in range(self.batch_size)
                ],
                dtype=torch.int32,
            ),
        )

        dense_features = torch.randn(
            self.batch_size,
            self.num_dense,
            generator=self.generator,
        )
        labels = torch.randint(
            low=0,
            high=2,
            size=(self.batch_size,),
            generator=self.generator,
        )

        batch = Batch(
            dense_features=dense_features,
            sparse_features=sparse_features,
            labels=labels,
        )
        return batch


class SyntheticRecDataset(IterableDataset[Batch]):
    """
    TODO: complete this doc
    Synthetic iterable dataset used to generate batches for recommender systems
    (RecSys). 

    Args:
        keys (List[str]): List of feature names for sparse features.
        batch_size (int): batch size.
        pooling_factor_per_feature (int): Number of IDs per sparse feature.
        num_embeddings_per_feature (Optional[List[int]]): Max sparse id value per feature in keys. Each
            sparse ID will be taken modulo the corresponding value from this argument.
        num_dense (int): Number of dense features.
        manual_seed (int): Seed for deterministic behavior.

    Example::
    """

    def __init__(
        self,
        keys: List[str],
        batch_size: int,
        pooling_factor_per_feature: List[int] = None,
        num_embeddings_per_feature: List[int] = None,
        num_dense: int = 50,
        manual_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.batch_generator = _SyntheticRecBatch(
            keys=keys,
            batch_size=batch_size,
            pooling_factor_per_feature=pooling_factor_per_feature,
            num_embeddings_per_feature=num_embeddings_per_feature,
            num_dense=num_dense,
            manual_seed=manual_seed,
        )

    def __iter__(self) -> Iterator[Batch]:
        return iter(self.batch_generator)
