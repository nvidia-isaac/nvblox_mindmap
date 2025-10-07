# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

from mindmap.geometry.pytorch3d_transforms import (
    quaternion_invert,
    quaternion_multiply,
    quaternion_to_axis_angle,
)

TRANS_LENGTH = 3


@dataclass
class LossWeights:
    pos_loss: float = 30.0
    rot_loss: float = 10.0
    gripper_loss: float = 1.0
    head_yaw_loss: float = 1.0


def compute_loss(
    pred: torch.Tensor,
    head_yaw_pred: torch.Tensor,
    target: torch.Tensor,
    gt_openess: torch.Tensor,
    gt_head_yaw: torch.Tensor,
    loss_weights: LossWeights,
    predict_head_yaw: bool,
    rotation_form: str = "quaternion",
) -> torch.Tensor:
    """Return L1 loss of the gripper position & rotation, BCE of gripper openess"""
    assert pred.shape[:-1] == target.shape[:-1]
    assert pred.shape[-1] == target.shape[-1] + gt_openess.shape[-1]

    pred_trans, pred_rot, pred_openess = destructure_action(pred, rotation_form=rotation_form)
    gt_trans, gt_rot, _ = destructure_action(target, rotation_form=rotation_form)

    pos_loss = F.l1_loss(pred_trans, gt_trans, reduction="mean")
    rot_loss = F.l1_loss(pred_rot, gt_rot, reduction="mean")
    gripper_loss = 0

    if torch.numel(gt_openess) > 0:
        gripper_loss = F.binary_cross_entropy_with_logits(pred_openess, gt_openess)

    total_loss = (
        loss_weights.pos_loss * pos_loss
        + loss_weights.rot_loss * rot_loss
        + loss_weights.gripper_loss * gripper_loss
    )

    detached_head_yaw_loss = None
    if predict_head_yaw:
        # NOTE(remos): We assume that the head yaw is in range [-pi, pi)
        #              and there is no wrap around at pi i.e. the head can not turn more than 180 degrees.
        #              Therefore we can treat it as linear range and use standard MSE loss.
        assert torch.all(gt_head_yaw >= -torch.pi) and torch.all(gt_head_yaw < torch.pi)
        head_yaw_loss = F.mse_loss(head_yaw_pred, gt_head_yaw, reduction="mean")
        total_loss += loss_weights.head_yaw_loss * head_yaw_loss
        detached_head_yaw_loss = head_yaw_loss.detach()

    # NOTE(remos): backprop only allowed on total loss.
    return (
        total_loss,
        pos_loss.detach(),
        rot_loss.detach(),
        gripper_loss.detach(),
        detached_head_yaw_loss,
    )


def compute_metrics(
    pred: torch.Tensor,
    head_yaw_pred: torch.Tensor,
    target: torch.Tensor,
    gt_head_yaw: torch.Tensor,
    predict_head_yaw: bool,
    rotation_form: str = "quaternion",
) -> torch.Tensor:
    """Return proxy metrics for checkpoint evaluation, mean distance(l2) of the gripper position,
    l1 distance of the gripper rotation."""
    assert pred.shape[:-1] == target.shape[:-1]
    pred_trans, pred_rot, pred_openness = destructure_action(pred, rotation_form=rotation_form)
    gt_trans, gt_rot, gt_openness = destructure_action(target, rotation_form=rotation_form)

    metrics = {}

    # Distance error
    distances_square = (pred_trans - gt_trans) ** 2
    distances_sqrt = distances_square.sqrt()
    distances_sse_sqrt = distances_square.sum(-1).sqrt()
    metrics["distance_m"] = torch.mean(distances_sse_sqrt)
    # equvalent to absolute error
    metrics["distance_m_x"] = torch.mean(distances_sqrt[..., 0])
    metrics["distance_m_y"] = torch.mean(distances_sqrt[..., 1])
    metrics["distance_m_z"] = torch.mean(distances_sqrt[..., 2])

    metrics["distance_m_std"] = torch.std(distances_sse_sqrt)
    metrics["distance_m_std_x"] = torch.std(distances_sqrt[..., 0])
    metrics["distance_m_std_y"] = torch.std(distances_sqrt[..., 1])
    metrics["distance_m_std_z"] = torch.std(distances_sqrt[..., 2])

    # Compute bias (mean error), not absolute error
    # Also now get the mean of the bias for each gripper
    biases = pred_trans - gt_trans
    metrics["bias"] = biases.mean(axis=(0, 1, 2))

    # Rotation accuracy
    # NOTE(alexmillane): l1 loss between two quaternions does not make any sense,
    # however I am keeping it in order to be able to compare new jobs to old jobs.
    # TODO(alexmillane): Remove this when these comparisons are no longer needed.
    metrics["rot_l1"] = (pred_rot - gt_rot).abs().sum(-1).mean()

    # Rotation error in degrees
    q_delta = quaternion_multiply(pred_rot, quaternion_invert(gt_rot))
    aa_delta = quaternion_to_axis_angle(q_delta)
    angle_delta = torch.norm(aa_delta, dim=-1)
    angle_delta_deg = angle_delta * 180 / torch.pi
    metrics["rot_error_deg"] = angle_delta_deg.mean()

    # gripper openness accuracy
    metrics["openness_l1"] = (pred_openness - gt_openness).abs().sum(-1).mean()

    # Head yaw accuracy
    if predict_head_yaw:
        metrics["head_yaw_error_deg"] = (head_yaw_pred - gt_head_yaw).abs().mean() * 180 / torch.pi

    return metrics


def destructure_action(
    action: torch.Tensor, rotation_form: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose action into gripper position, rotation and gripper openness if applicable"""
    if rotation_form == "quaternion":
        rot_length = 4
    elif rotation_form == "6D":
        rot_length = 6
    else:
        raise NotImplementedError
    assert action.ndim >= 2
    assert action.shape[-1] >= TRANS_LENGTH + rot_length
    assert action.shape[-1] <= TRANS_LENGTH + rot_length + 1

    if action.shape[-1] == TRANS_LENGTH + rot_length:
        openess = None
    else:
        openess = action[..., TRANS_LENGTH + rot_length :]

    return (
        action[..., :TRANS_LENGTH],
        action[..., TRANS_LENGTH : TRANS_LENGTH + rot_length],
        openess,
    )
