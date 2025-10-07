# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import torch

from mindmap.embodiments.arm.robot_state import ArmEmbodimentRobotState
from mindmap.embodiments.humanoid.action import HumanoidEmbodimentAction
from mindmap.embodiments.humanoid.robot_state import HumanoidEmbodimentRobotState


def test_arm_embodiment_state():
    # Dummy value
    state = ArmEmbodimentRobotState(
        W_t_W_Eef=torch.tensor([1.0, 2.0, 3.0]),
        q_wxyz_W_Eef=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        gripper_jaw_positions=torch.tensor([0.25, 0.75]),
    )

    # Test to tensor
    tensor = state.to_tensor()
    assert tensor[0] == 1.0
    assert tensor[1] == 2.0
    assert tensor[2] == 3.0
    assert tensor[3] == 1.0
    assert tensor[4] == 0.0
    assert tensor[5] == 0.0
    assert tensor[6] == 0.0
    assert tensor[7] == 0.25
    assert tensor[8] == 0.75

    assert tensor.shape == (ArmEmbodimentRobotState.state_size(),)

    # Test from tensor (round trip)
    state_from_tensor = ArmEmbodimentRobotState.from_tensor(tensor)
    assert state_from_tensor.W_t_W_Eef.equal(state.W_t_W_Eef)
    assert state_from_tensor.q_wxyz_W_Eef.equal(state.q_wxyz_W_Eef)
    assert state_from_tensor.gripper_jaw_positions.equal(state.gripper_jaw_positions)


def test_humanoid_embodiment_state():
    # Dummy value
    state = HumanoidEmbodimentRobotState(
        W_t_W_LeftEef=torch.tensor([1.0, 2.0, 3.0]),
        q_wxyz_W_LeftEef=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        left_hand_joint_states=torch.arange(0, 11).float(),
        W_t_W_RightEef=torch.tensor([1.0, 2.0, 3.0]),
        q_wxyz_W_RightEef=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        right_hand_joint_states=torch.arange(0, 11).float(),
        head_yaw_rad=torch.tensor([2.0]),
    )

    # Test to tensor
    tensor = state.to_tensor()
    assert torch.all(tensor[0:3] == torch.tensor([1.0, 2.0, 3.0]))
    assert torch.all(tensor[3:7] == torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.all(tensor[7:18] == torch.arange(0, 11).float())
    assert torch.all(tensor[18:21] == torch.tensor([1.0, 2.0, 3.0]))
    assert torch.all(tensor[21:25] == torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.all(tensor[25:36] == torch.arange(0, 11).float())
    assert torch.all(tensor[36] == 2.0)

    assert tensor.shape == (HumanoidEmbodimentRobotState.state_size(),)

    # Test from tensor (round trip)
    state_from_tensor = HumanoidEmbodimentRobotState.from_tensor(tensor)
    assert state_from_tensor.W_t_W_LeftEef.equal(state.W_t_W_LeftEef)
    assert state_from_tensor.q_wxyz_W_LeftEef.equal(state.q_wxyz_W_LeftEef)
    assert state_from_tensor.left_hand_joint_states.equal(state.left_hand_joint_states)
    assert state_from_tensor.W_t_W_RightEef.equal(state.W_t_W_RightEef)
    assert state_from_tensor.q_wxyz_W_RightEef.equal(state.q_wxyz_W_RightEef)
    assert state_from_tensor.right_hand_joint_states.equal(state.right_hand_joint_states)


def test_history_from_tensor():
    history_length = 10

    # ArmEmbodimentRobotState
    tensor = torch.randn(1, history_length, ArmEmbodimentRobotState.state_size())
    states = ArmEmbodimentRobotState.history_from_tensor(tensor)
    for state_idx, state in enumerate(states):
        assert torch.all(state.to_tensor() == tensor[0, state_idx, :])

    # HumanoidEmbodimentRobotState
    tensor = torch.randn(1, history_length, HumanoidEmbodimentRobotState.state_size())
    states = HumanoidEmbodimentRobotState.history_from_tensor(tensor)
    for state_idx, state in enumerate(states):
        assert torch.all(state.to_tensor() == tensor[0, state_idx, :])


def test_humanoid_hand_joint_states_tensor():
    # The hand joint actions have a weird ordering.
    # This test checks that when we convert the action object to a tensor
    # we get the joint states in the final 22 elements of the tensor.
    # This is a basic sanity check but it's the best I can think of for now.

    # Test action:
    # - pose elements are all zero
    # - hand joint states are all one.
    action = HumanoidEmbodimentAction(
        W_t_W_LeftEef=torch.zeros((3,)),
        q_wxyz_W_LeftEef=torch.zeros((4,)),
        left_hand_joint_states=torch.ones((11,)),
        W_t_W_RightEef=torch.zeros((3,)),
        q_wxyz_W_RightEef=torch.zeros((4,)),
        right_hand_joint_states=torch.ones((11,)),
        head_yaw_rad=torch.tensor([2.0]),
    )
    # Convert to tensor
    action_tensor = action.to_tensor(include_head_yaw=True)
    assert action_tensor.shape == torch.Size((3 + 4 + 11 + 3 + 4 + 11 + 1,))
    # Check that the hand joint states are all one.
    hand_action_tensor = action_tensor[-22:]
    assert hand_action_tensor.shape == torch.Size((11 + 11,))
    assert torch.all(hand_action_tensor == torch.ones((22,)))
    # Check that the pose elements are all zero.
    eef_pose_tensor = action_tensor[:-23]
    assert eef_pose_tensor.shape == torch.Size((3 + 4 + 3 + 4,))
    assert torch.all(eef_pose_tensor == 0.0)
    # Check that the head yaw is 2.0
    head_yaw_tensor = action_tensor[-23:-22]
    assert head_yaw_tensor.shape == torch.Size((1,))
    assert torch.all(head_yaw_tensor == 2.0)
