# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import copy
import os

import pytest
import torch

from mindmap.diffuser_actor.layers import FFWRelativeCrossAttentionModule

EMBEDDING_DIM = 120
NUM_ATTN_HEADS = int(EMBEDDING_DIM / 6)
NUM_LAYERS = 2
BATCH_SIZE = 12
NUM_QUERIES = 3
NUM_VALUES = 123
N_POINTS = 51

MASK_INDEX = 1


def test_weight_masking():
    """Test that assumptions we make about masking in multihead custom attention are correct."""
    cross_attn = FFWRelativeCrossAttentionModule(
        embedding_dim=EMBEDDING_DIM,
        num_attn_heads=NUM_ATTN_HEADS,
        num_layers=NUM_LAYERS,
    ).eval()

    query = torch.randn(NUM_QUERIES, BATCH_SIZE, EMBEDDING_DIM)
    value = torch.randn(NUM_VALUES, BATCH_SIZE, EMBEDDING_DIM)

    query_pos = torch.randn(BATCH_SIZE, NUM_QUERIES, EMBEDDING_DIM, 2)
    value_pos = torch.randn(BATCH_SIZE, NUM_VALUES, EMBEDDING_DIM, 2)

    time_embeddings = torch.randn(BATCH_SIZE, EMBEDDING_DIM)

    # Create output without any masking or any modification
    original_output, original_weights = cross_attn(
        query=query,
        value=value,
        query_pos=query_pos,
        value_pos=value_pos,
        diff_ts=time_embeddings,
    )

    # Let's first ensure that the output from two identical runs are deterministic
    original_output_2, original_weights_2 = cross_attn(
        query=query,
        value=value,
        query_pos=query_pos,
        value_pos=value_pos,
        diff_ts=time_embeddings,
    )
    assert torch.all(torch.isclose(original_output[-1], original_output_2[-1]))
    assert torch.all(torch.isclose(original_weights[-1], original_weights_2[-1]))

    # TEST WITH A MASK THAT DOESN'T EXCLUDE ANYTHING.
    # This should not change the output
    mask_no_exclusion = torch.zeros(BATCH_SIZE, NUM_VALUES, dtype=torch.bool)
    mask_no_exclusion_output, mask_no_exclusion_weights = cross_attn(
        query=query,
        value=value,
        query_pos=query_pos,
        value_pos=value_pos,
        diff_ts=time_embeddings,
        key_padding_mask=mask_no_exclusion,
    )
    assert torch.all(torch.isclose(original_output[-1], mask_no_exclusion_output[-1]))
    assert torch.all(torch.isclose(original_weights[-1], mask_no_exclusion_weights[-1]))

    # Modify some positions and values
    indices_to_modify = torch.randint(low=0, high=N_POINTS, size=(int(N_POINTS / 10),))
    modified_mask = torch.zeros(BATCH_SIZE, NUM_VALUES, dtype=torch.bool)
    modified_mask[:, indices_to_modify] = True
    value_pos_modified = copy.deepcopy(value_pos)
    value_pos_modified[modified_mask] *= 0.1

    value_modified = copy.deepcopy(value)
    value_modified[modified_mask.transpose(0, 1)] *= 0.1

    # RUN WITH MODIFIED POSITIONS BUT WITHOUT MASKING.
    # Output is expected to change
    modified_output, modified_weights = cross_attn(
        query=query,
        value=value_modified,
        query_pos=query_pos,
        value_pos=value_pos_modified,
        diff_ts=time_embeddings,
    )
    assert not torch.all(torch.isclose(original_output[-1], modified_output[-1]))
    assert not torch.all(torch.isclose(original_weights[-1], modified_weights[-1]))

    # RUN WITH MASKING WITH AND WITHOUT MODIFIED POSITIONS.
    # Output is expected to be the same between the two, since the masked positions will be removed from attention.
    masked_output1, masked_weights1 = cross_attn(
        query=query,
        value=value_modified,
        query_pos=query_pos,
        value_pos=value_pos_modified,
        diff_ts=time_embeddings,
        key_padding_mask=modified_mask,
    )
    masked_output2, masked_weights2 = cross_attn(
        query=query,
        value=value,
        query_pos=query_pos,
        value_pos=value_pos,
        diff_ts=time_embeddings,
        key_padding_mask=modified_mask,
    )
    assert torch.all(torch.isclose(masked_output1[-1], masked_output2[-1]))
    assert torch.all(torch.isclose(masked_weights1[-1], masked_weights2[-1]))
