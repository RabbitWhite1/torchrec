#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from functools import reduce
from typing import Tuple, Dict, Optional, List, cast, Union

import torch
import torch.distributed as dist
from torch import nn
from torchrec.distributed.collective_utils import (
    invoke_on_rank_and_broadcast_result,
)
from torchrec.distributed.planner.constants import MAX_SIZE
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator, get_partition_by_type, calculate_shard_sizes_and_offsets
from torchrec.distributed.planner.partitioners import GreedyPerfPartitioner
from torchrec.distributed.planner.perf_models import NoopPerfModel
from torchrec.distributed.planner.planners import _to_sharding_plan
from torchrec.distributed.planner.proposers import GreedyProposer, UniformProposer
from torchrec.distributed.planner.stats import EmbeddingStats
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.planner.types import (
    ParameterConstraints,
    Partitioner,
    Topology,
    Stats,
    Shard,
    Storage,
    ShardingOption,
    StorageReservation,
    Enumerator,
    Proposer,
    PerfModel,
    PlannerError,
)
from torchrec.distributed.planner.utils import sharder_name
from torchrec.distributed.types import (
    EnumerableShardingSpec,
    ShardMetadata,
)
from torchrec.distributed.types import (
    ShardingPlan,
    ShardingPlanner,
    ModuleSharder,
    ShardingType,
    ParameterSharding,
)

import rich
import random
import numpy as np

def calculate_shard_sizes_and_offsets_for_row_cache(tensor, world_size, cache_rows=64):
    (rows, columns) = tensor.shape
    assert world_size >= 2
    if rows <= cache_rows:
        assert False, "should not happen"
        return ([rows], [columns])
    else:
        return (
            [[cache_rows, columns], [rows-cache_rows, columns]] + [[0, 0]] * (world_size - 2), 
            [[0, 0], [cache_rows, 0]] + [[rows, 0]] * (world_size - 2)
        )


class MyShardingPlanner(ShardingPlanner):
    def __init__(
        self,
        feature_name_to_rank: Dict[str, int],
        feature_name_to_pooling_factor: Dict[str, int],
        topology: Topology
    ) -> None:
        self._topology = topology
        self._feature_name_to_rank = feature_name_to_rank
        self._feature_name_to_pooling_factor = feature_name_to_pooling_factor

    def collective_plan(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
        # pyre-fixme[11]: Annotation `ProcessGroup` is not defined as a type.
        pg: dist.ProcessGroup,
    ) -> ShardingPlan:
        """
        Call self.plan(...) on rank 0 and broadcast
        """
        return invoke_on_rank_and_broadcast_result(
            pg,
            0,
            self.plan,
            module,
            sharders,
        )

    def plan(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> ShardingPlan:
        sharder_map: Dict[str, ModuleSharder[nn.Module]] = {
            sharder_name(sharder.module_type): sharder for sharder in sharders
        }

        my_plan: List[ShardingOption] = []
        for child_path, child_module in module.named_modules():
            sharder_key = sharder_name(type(child_module))
            sharder = sharder_map.get(sharder_key, None)
            if not sharder:
                continue
            
            for name, param in sharder.shardable_parameters(child_module).items():
                sharding_type = ShardingType.TABLE_WISE.value
                # sharding_type = ShardingType.ROW_WISE.value
                if sharding_type == ShardingType.TABLE_WISE.value:
                    shard_sizes, shard_offsets = calculate_shard_sizes_and_offsets(
                        tensor=param,
                        world_size=self._topology.world_size,
                        local_world_size=self._topology.local_world_size,
                        sharding_type=sharding_type,
                        col_wise_shard_dim=None,
                    )
                    shard_ranks = [self._feature_name_to_rank[name]] * len(shard_sizes)
                else:
                    assert sharding_type == ShardingType.ROW_WISE.value
                    shard_sizes, shard_offsets = calculate_shard_sizes_and_offsets_for_row_cache(
                        tensor=param,
                        world_size=self._topology.world_size,
                    )
                    this_rank = self._feature_name_to_rank[name]
                    other_ranks = list(range(0, self._topology.world_size))
                    other_ranks.remove(this_rank)
                    other_ranks = np.random.permutation(other_ranks).tolist()
                    shard_ranks = [this_rank] + other_ranks
                    if not (len(shard_sizes) == len(shard_offsets) == len(shard_ranks)):
                        raise RuntimeError("lengths not match")
                my_plan.append(
                    ShardingOption(
                        name=name,
                        tensor=param,
                        module=(child_path, child_module),
                        upstream_modules=[],
                        downstream_modules=[],
                        input_lengths=[1.0],#input_lengths,
                        batch_size=self._topology.batch_size,
                        compute_kernel='batched_fused', #compute_kernel,
                        sharding_type=sharding_type,
                        partition_by=get_partition_by_type(sharding_type),
                        shards=[
                            Shard(size=size, offset=offset, rank=rank)
                            for size, offset, rank in zip(shard_sizes, shard_offsets, shard_ranks)
                        ],
                    )
                )
        sharding_plan = _to_sharding_plan(my_plan, self._topology)
        
        return sharding_plan
