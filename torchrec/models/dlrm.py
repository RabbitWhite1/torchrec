#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Dict

import torch
import torch.distributed as dist
from torch import nn
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.mlp import MLP
from torchrec.sparse.jagged_tensor import (
    KeyedJaggedTensor,
    KeyedTensor,
)
from torchrec.test_utils.timer import Timer


def choose(n: int, k: int) -> int:
    """
    Simple implementation of math.comb for python 3.7 compatibility.
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


class SparseArch(nn.Module):
    """
    Processes the sparse features of DLRM. Does embedding lookups for all EmbeddingBag
    and embedding features of each collection.

    Args:
        embedding_bag_collection (EmbeddingBagCollection): represents a
            collection of pooled embeddings

    Example::

        eb1_config = EmbeddingBagConfig(
           name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
           name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )
        ebc_config = EmbeddingBagCollectionConfig(tables=[eb1_config, eb2_config])

        ebc = EmbeddingBagCollection(config=ebc_config)
        sparse_arch = SparseArch(embedding_bag_collection)

        #     0       1        2  <-- batch
        # 0   [0,1] None    [2]
        # 1   [3]    [4]    [5,6,7]
        # ^
        # feature
        features = KeyedJaggedTensor.from_offsets_sync(
           keys=["f1", "f2"],
           values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
           offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        sparse_embeddings = sparse_arch(features)
    """

    def __init__(self, embedding_bag_collection: EmbeddingBagCollection) -> None:
        super().__init__()
        self.embedding_bag_collection: EmbeddingBagCollection = embedding_bag_collection
        assert (
            self.embedding_bag_collection.embedding_bag_configs
        ), "Embedding bag collection cannot be empty!"
        self.D: int = self.embedding_bag_collection.embedding_bag_configs[
            0
        ].embedding_dim
        self._sparse_feature_names: List[str] = [
            name
            for conf in embedding_bag_collection.embedding_bag_configs
            for name in conf.feature_names
        ]

        self.F: int = len(self._sparse_feature_names)

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> KeyedTensor:
        """
        Args:
            features (KeyedJaggedTensor): an input tensor of sparse features.

        Returns:
            torch.Tensor of shape B X F X D
        """
        sparse_features: KeyedTensor = self.embedding_bag_collection(features)
        return sparse_features

        # Below are moved to interaction, because I want to support different feature dim.
        # If put here, we cannot simply do the reshape(B, self.F, self.D)
        """B: int = features.stride()

        sparse: Dict[str, torch.Tensor] = sparse_features.to_dict()
        sparse_values: List[torch.Tensor] = []
        for name in self.sparse_feature_names:
            sparse_values.append(sparse[name])

        return torch.cat(sparse_values, dim=1).reshape(B, self.F, self.D)"""

    @property
    def sparse_feature_names(self) -> List[str]:
        return self._sparse_feature_names


class DenseArch(nn.Module):
    """
    Processes the dense features of DLRM model.

    Args:
        in_features (int): dimensionality of the dense input features.
        layer_sizes (List[int]): list of layer sizes.
        device (Optional[torch.device]): default compute device.

    Example::

        B = 20
        D = 3
        dense_arch = DenseArch(10, layer_sizes=[15, D])
        dense_embedded = dense_arch(torch.rand((B, 10)))
    """

    def __init__(
        self,
        in_features: int,
        layer_sizes: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.model: nn.Module = MLP(
            in_features, layer_sizes, bias=True, activation="relu", device=device
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): an input tensor of dense features.

        Returns:
            torch.Tensor: an output tensor of size B X D
        """
        return self.model(features)


class InteractionArch(nn.Module):
    """
    Processes the output of both `SparseArch` (sparse_features) and `DenseArch`
    (dense_features). Returns the pairwise dot product of each sparse feature pair,
    the dot product of each sparse features with the output of the dense layer,
    and the dense layer itself (all concatenated).

    .. note::
        The dimensionality of the `dense_features` (D) is expected to match the
        dimensionality of the `sparse_features` so that the dot products between them can be
        computed.


    Args:
        num_sparse_features : int = F

    Example::

        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        inter_arch = InteractionArch(num_sparse_features=len(keys))

        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))

        #  B X (D + F + F choose 2)
        concat_dense = inter_arch(dense_features, sparse_features)
    """

    def __init__(self, num_sparse_features: int) -> None:
        super().__init__()
        self.F: int = num_sparse_features
        self.triu_indices: torch.Tensor = torch.triu_indices(
            self.F + 1, self.F + 1, offset=1
        )

    def forward(
        self, dense_features: torch.Tensor, sparse_features: KeyedTensor
    ) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): an input tensor of size B X D.
            sparse_features (KeyedTensor): an input KeyedTensor with batch_size as B, 
                num_features as F, all feature_dim as D, 

        Returns:
            torch.Tensor: an output tensor of size B X (D + F + F choose 2).
        """
        if self.F <= 0:
            return dense_features
        (B, D) = dense_features.shape

        rank = dist.get_rank()
        # moved to here from SparseArch
        with Timer(rank, "inter_arch_step1"):
            sparse_features_dict: Dict[str, torch.Tensor] = sparse_features.to_dict()
            sparse_values: List[torch.Tensor] = []
            for sparse_value in sparse_features_dict.values():
                if sparse_value.shape[1] != D:
                    raise ValueError(f"sparse feature dim should be all the same as dense feature dim {D}")
                sparse_values.append(sparse_value)

            # sparse_features = torch.cat(sparse_values, dim=1).reshape(B, self.F, D)

            combined_values = torch.cat(
                (dense_features.unsqueeze(1), *sparse_values), dim=1
            )

        # dense/sparse + sparse/sparse interaction
        # size B X (F + F choose 2)
        with Timer(rank, "inter_arch_step2"):
            interactions = torch.bmm(
                combined_values, torch.transpose(combined_values, 1, 2)
            )
            interactions_flat = interactions[:, self.triu_indices[0], self.triu_indices[1]]

        with Timer(rank, "inter_arch_step3"):
            return torch.cat((dense_features, interactions_flat), dim=1)

    @property
    def sparse_feature_names(self) -> List[str]:
        return self._sparse_feature_names


class CatInteractionArch(nn.Module):
    """
    Processes the output of both `SparseArch` (sparse_features) and `DenseArch`
    (dense_features). Returns the concatenation of each sparse feature pair, and 
    the dense layer itself (all concatenated).

    .. note::
        D: `dense_feature_dim`


    Args:
        embedding_dim_per_feature (List[int])
    """

    def __init__(self, embedding_dim_per_feature: List[int]) -> None:
        super().__init__()
        self._embedding_dim_per_feature = embedding_dim_per_feature
        self._done = False

    def forward(
        self, dense_features: torch.Tensor, sparse_features: KeyedTensor
    ) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): an input tensor of size B X D.
            sparse_features (torch.Tensor): an input tensor of size B X F X D

        Returns:
            torch.Tensor: an output tensor of size B X (D + F + F choose 2).
        """
        rank = dist.get_rank()
        # moved to here from SparseArch
        with Timer(rank, "inter_arch_step1"):
            sparse_features_dict: Dict[str, torch.Tensor] = sparse_features.to_dict()
            sparse_values: List[torch.Tensor] = []
            for sparse_value in sparse_features_dict.values():
                sparse_values.append(sparse_value)

        with Timer(rank, "inter_arch_step2"):
            return torch.cat((dense_features, *sparse_values), dim=1)


class OverArch(nn.Module):
    """
    Final Arch of DLRM - simple MLP over OverArch.

    Args:
        in_features (int): size of the input.
        layer_sizes (List[int]): sizes of the layers of the `OverArch`.
        device (Optional[torch.device]): default compute device.

    Example::

        B = 20
        D = 3
        over_arch = OverArch(10, [5, 1])
        logits = over_arch(torch.rand((B, 10)))
    """

    def __init__(
        self,
        in_features: int,
        layer_sizes: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if len(layer_sizes) <= 1:
            raise ValueError("OverArch must have multiple layers.")
        self.model: nn.Module = nn.Sequential(
            MLP(
                in_features,
                layer_sizes[:-1],
                bias=True,
                activation="relu",
                device=device,
            ),
            nn.Linear(layer_sizes[-2], layer_sizes[-1], bias=True, device=device),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: torch.Tensor

        Returns:
            torch.Tensor  - size B X layer_sizes[-1]
        """
        return self.model(features)


class DLRM(nn.Module):
    """
    Recsys model from "Deep Learning Recommendation Model for Personalization and
    Recommendation Systems" (https://arxiv.org/abs/1906.00091). Processes sparse
    features by learning pooled embeddings for each feature. Learns the relationship
    between dense features and sparse features by projecting dense features into the
    same embedding space. Also, learns the pairwise relationships between sparse
    features.

    The module assumes all sparse features have the same embedding dimension
    (i.e. each EmbeddingBagConfig uses the same embedding_dim).

    The following notation is used throughout the documentation for the models:

    * F: number of sparse features
    * D: embedding_dimension of sparse features
    * B: batch size
    * num_features: number of dense features

    Args:
        embedding_bag_collection (EmbeddingBagCollection): collection of embedding bags
            used to define `SparseArch`.
        dense_in_features (int): the dimensionality of the dense input features.
        dense_arch_layer_sizes (List[int]): the layer sizes for the `DenseArch`.
        over_arch_layer_sizes (List[int]): the layer sizes for the `OverArch`.
            The output dimension of the `InteractionArch` should not be manually specified here.
        dense_device (Optional[torch.device]): default compute device.
        interaction_type (str): 'real' or 'cat'

    Example::

        B = 2
        D = 8

        eb1_config = EmbeddingBagConfig(
           name="t1", embedding_dim=D, num_embeddings=100, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
           name="t2",
           embedding_dim=D,
           num_embeddings=100,
           feature_names=["f2"],
        )
        ebc_config = EmbeddingBagCollectionConfig(tables=[eb1_config, eb2_config])

        ebc = EmbeddingBagCollection(config=ebc_config)
        model = DLRM(
           embedding_bag_collection=ebc,
           dense_in_features=100,
           dense_arch_layer_sizes=[20],
           over_arch_layer_sizes=[5, 1],
        )

        features = torch.rand((B, 100))

        #     0       1
        # 0   [1,2] [4,5]
        # 1   [4,3] [2,9]
        # ^
        # feature
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
           keys=["f1", "f3"],
           values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]),
           offsets=torch.tensor([0, 2, 4, 6, 8]),
        )

        logits = model(
           dense_features=features,
           sparse_features=sparse_features,
        )
    """

    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
        dense_device: Optional[torch.device] = None,
        interaction_type: str = 'real'
    ) -> None:
        super().__init__()
        assert (
            len(embedding_bag_collection.embedding_bag_configs) > 0
        ), "At least one embedding bag is required"
        dense_arch_out_dim = dense_arch_layer_sizes[-1]

        # sparse arch
        self.sparse_arch: SparseArch = SparseArch(embedding_bag_collection)
        num_sparse_features: int = len(self.sparse_arch.sparse_feature_names)

        # dense arch
        self.dense_arch = DenseArch(
            in_features=dense_in_features,
            layer_sizes=dense_arch_layer_sizes,
            device=dense_device,
        )
        # interaction arch
        if interaction_type == 'real':
            for i in range(1, len(embedding_bag_collection.embedding_bag_configs)):
                conf_prev = embedding_bag_collection.embedding_bag_configs[i - 1]
                conf = embedding_bag_collection.embedding_bag_configs[i]
                assert (
                    conf_prev.embedding_dim == conf.embedding_dim
                ), "All EmbeddingBagConfigs must have the same dimension"
            embedding_dim: int = embedding_bag_collection.embedding_bag_configs[
                0
            ].embedding_dim
            if dense_arch_out_dim != embedding_dim:
                raise ValueError(
                    f"embedding_bag_collection dimension ({embedding_dim}) and final dense "
                    f"arch layer size ({dense_arch_out_dim=}) must match."
                )
            self.inter_arch = InteractionArch(num_sparse_features=num_sparse_features)

            over_in_features: int = (
                embedding_dim + choose(num_sparse_features, 2) + num_sparse_features
            )
        elif interaction_type == 'cat':
            embedding_dim_per_feature = [
                cfg.embedding_dim for cfg in embedding_bag_collection.embedding_bag_configs
            ]
            self.inter_arch = CatInteractionArch(embedding_dim_per_feature=embedding_dim_per_feature)

            over_in_features: int = dense_arch_out_dim + sum(embedding_dim_per_feature)
        # over arch
        self.over_arch = OverArch(
            in_features=over_in_features,
            layer_sizes=over_arch_layer_sizes,
            device=dense_device,
        )

    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor):
            sparse_features (KeyedJaggedTensor):

        Returns:
            torch.Tensor:
        """
        rank = dist.get_rank()
        with Timer(rank, "dense_arch"):
            embedded_dense = self.dense_arch(dense_features)
        with Timer(rank, "sparse_arch"):
            embedded_sparse = self.sparse_arch(sparse_features)
        with Timer(rank, "inter_arch"):
            concatenated_dense = self.inter_arch(
                dense_features=embedded_dense, sparse_features=embedded_sparse
            )
        with Timer(rank, "over_arch"):
            logits = self.over_arch(concatenated_dense)
        return logits
