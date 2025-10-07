# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import numpy as np
import torch

from mindmap.embodiments.arm.keypose_estimation import ArmEmbodimentKeyposeEstimator
from mindmap.embodiments.arm.robot_state import ArmEmbodimentRobotState
from mindmap.keyposes.keypose_detection_mode import KeyposeDetectionMode

GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.01


def _add_gripper_event(index, gripper_pos, num_samples, event):
    assert event in ["open", "close"]
    num_samples = 5
    gripper_pos[index:] = GRIPPER_CLOSED if event == "close" else GRIPPER_OPEN

    for i in range(0, num_samples):
        idx = (num_samples - i) if event == "close" else i
        val = GRIPPER_CLOSED + (GRIPPER_OPEN - GRIPPER_CLOSED) * idx / num_samples
        gripper_pos[index + i] = val


def test_keypose_detection():
    # Now this test might be only for the arm embodiment.
    sequence_length = 100

    # Define ground-truth indices
    gt_zpeak = 25
    gt_close_event_start = 10
    gt_open_event_start = 40
    gt_event_num_samples = 5

    # Generate gripper position events
    gripper_pos = np.ones((sequence_length, 2)) * GRIPPER_OPEN
    _add_gripper_event(gt_close_event_start, gripper_pos, gt_event_num_samples, "close")
    _add_gripper_event(gt_open_event_start, gripper_pos, gt_event_num_samples, "open")

    # Genrate eef trajectory with one detectable peak in Z
    eef_pos = np.zeros((sequence_length, 3))
    eef_pos[gt_zpeak, 2] = 1

    eef_quat = np.zeros((sequence_length, 4))
    eef_quat[:, 3] = 1

    # First and third frame around grasp event should be keypose
    extra_keyposes_around_grasp_events = [1, 3]

    # Create a robot state with the gripper position and eef position
    # Gripper state is world position, world orientation, gripper position
    gripper_state = torch.cat(
        (torch.from_numpy(eef_pos), torch.from_numpy(eef_quat), torch.from_numpy(gripper_pos)),
        dim=1,
    )

    arm_robot_state_class = ArmEmbodimentRobotState
    gripper_states = []
    for i in range(gripper_state.shape[0]):
        gripper_states.append(arm_robot_state_class.from_tensor(gripper_state[i]))

    arm_keypose_estimator = ArmEmbodimentKeyposeEstimator()

    # Run keypose detection
    keypose_indices = arm_keypose_estimator.extract_keypose_indices(
        gripper_states,
        extra_keyposes_around_grasp_events,
        KeyposeDetectionMode.HIGHEST_Z_BETWEEN_GRASP,
    )
    assert np.all(keypose_indices == sorted(keypose_indices))

    # check that open and close events were detected
    assert gt_close_event_start in keypose_indices
    assert gt_close_event_start + gt_event_num_samples + 1 in keypose_indices

    assert gt_open_event_start in keypose_indices
    assert gt_open_event_start + gt_event_num_samples + 1 in keypose_indices

    assert gt_zpeak in keypose_indices

    # Check that extra keyposes were detected
    for index in extra_keyposes_around_grasp_events:
        assert gt_open_event_start - index in keypose_indices
        assert gt_open_event_start + gt_event_num_samples + index + 1 in keypose_indices
        assert gt_close_event_start - index in keypose_indices
        assert gt_close_event_start + gt_event_num_samples + index + 1 in keypose_indices

    # Convert to policy states
    from mindmap.embodiments.arm.estimator import ArmEmbodimentOfflineEstimator

    offline_estimator = ArmEmbodimentOfflineEstimator()
    policy_states = offline_estimator.policy_states_from_robot_states(
        robot_state_vec=gripper_states, use_keyposes=True
    )

    # # Gripper is open until end of close event
    for gripper_policy_state in policy_states[: gt_close_event_start + gt_event_num_samples]:
        assert gripper_policy_state.gripper_closedness == torch.tensor(0.0)

    # Gripper is closed until beginning of open event
    for gripper_policy_state in policy_states[
        gt_close_event_start + gt_event_num_samples : gt_open_event_start + 1
    ]:
        assert gripper_policy_state.gripper_closedness == torch.tensor(1.0)

    # Gripper stays open until the end
    for gripper_policy_state in policy_states[gt_open_event_start + 1 :]:
        assert gripper_policy_state.gripper_closedness == torch.tensor(0.0)


if __name__ == "__main__":
    test_keypose_detection()
