# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from mindmap.cli.args import ClosedLoopAppArgs
from mindmap.closed_loop.gt_goals import get_gt_goals
from mindmap.embodiments.arm.embodiment import ArmEmbodiment
from mindmap.embodiments.embodiment_base import EmbodimentType
from mindmap.embodiments.humanoid.embodiment import HumanoidEmbodiment
from mindmap.embodiments.state_base import PolicyStateBase
from mindmap.embodiments.task_to_embodiment import get_embodiment_type_from_task
from mindmap.tasks.tasks import Tasks


def create_embodiment(task: Tasks, device: str = "cuda"):
    """Create the appropriate embodiment based on the task."""
    embodiment_type = get_embodiment_type_from_task(task)

    if embodiment_type == EmbodimentType.ARM:
        return ArmEmbodiment(device=device)
    elif embodiment_type == EmbodimentType.HUMANOID:
        return HumanoidEmbodiment(task, device=device)
    else:
        raise ValueError(f"Unsupported embodiment type: {embodiment_type}")


def extract_keypose_positions(
    policy_states: List[PolicyStateBase], embodiment_type: EmbodimentType
):
    """Extract 3D positions of keyposes from policy states and gripper closedness information."""
    positions = {}

    if embodiment_type == EmbodimentType.ARM:
        # Single end-effector for ARM
        arm_positions = []
        arm_closedness = []
        for state in policy_states:
            pos = state.W_t_W_Eef.cpu().numpy()
            closedness = bool(state.gripper_closedness.cpu().numpy())
            arm_positions.append(pos)
            arm_closedness.append(closedness)
        positions["arm"] = np.array(arm_positions)
        positions["arm_closedness"] = np.array(arm_closedness)

    elif embodiment_type == EmbodimentType.HUMANOID:
        # Two end-effectors for HUMANOID
        left_positions = []
        right_positions = []
        left_closedness = []
        right_closedness = []
        for state in policy_states:
            left_pos = state.W_t_W_LeftEef.cpu().numpy()
            right_pos = state.W_t_W_RightEef.cpu().numpy()
            left_close = bool(state.left_hand_closedness.cpu().numpy())
            right_close = bool(state.right_hand_closedness.cpu().numpy())
            left_positions.append(left_pos)
            right_positions.append(right_pos)
            left_closedness.append(left_close)
            right_closedness.append(right_close)
        positions["left"] = np.array(left_positions)
        positions["right"] = np.array(right_positions)
        positions["left_closedness"] = np.array(left_closedness)
        positions["right_closedness"] = np.array(right_closedness)

    return positions


def visualize_keyposes_3d(
    positions: dict,
    embodiment_type: EmbodimentType,
    demo_name: str,
    trajectory_positions: dict = None,
):
    """Create 3D visualization of keypose trajectories with gripper state visualization."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    if embodiment_type == EmbodimentType.ARM:
        # Single arm visualization
        arm_pos = positions["arm"]
        arm_closedness = positions["arm_closedness"]

        # Plot keypose trajectory in blue
        ax.plot(
            arm_pos[:, 0],
            arm_pos[:, 1],
            arm_pos[:, 2],
            c="blue",
            alpha=0.7,
            linewidth=2,
            label="Arm EEF Keypose Trajectory",
        )

        # Plot full trajectory with dashed line in same color
        if trajectory_positions is not None:
            full_arm_pos = trajectory_positions["arm"]
            ax.plot(
                full_arm_pos[:, 0],
                full_arm_pos[:, 1],
                full_arm_pos[:, 2],
                c="blue",
                alpha=0.5,
                linewidth=1,
                linestyle="--",
                label="Arm EEF Full Trajectory",
            )

        arm_open_mask = ~arm_closedness
        arm_closed_mask = arm_closedness

        # Green for open gripper
        if np.any(arm_open_mask):
            ax.scatter(
                arm_pos[arm_open_mask, 0],
                arm_pos[arm_open_mask, 1],
                arm_pos[arm_open_mask, 2],
                c="green",
                marker="o",
                s=80,
                label="Open Gripper Keyposes",
                edgecolors="black",
                linewidth=1,
            )

        # Red for closed gripper
        if np.any(arm_closed_mask):
            ax.scatter(
                arm_pos[arm_closed_mask, 0],
                arm_pos[arm_closed_mask, 1],
                arm_pos[arm_closed_mask, 2],
                c="red",
                marker="o",
                s=80,
                label="Closed Gripper Keyposes",
                edgecolors="black",
                linewidth=1,
            )

    elif embodiment_type == EmbodimentType.HUMANOID:
        # Dual arm visualization
        left_pos = positions["left"]
        right_pos = positions["right"]
        left_closedness = positions["left_closedness"]
        right_closedness = positions["right_closedness"]

        # Plot keypose trajectories
        ax.plot(
            left_pos[:, 0],
            left_pos[:, 1],
            left_pos[:, 2],
            c="blue",
            alpha=0.7,
            linewidth=2,
            label="Left EEF Keypose Trajectory",
        )
        ax.plot(
            right_pos[:, 0],
            right_pos[:, 1],
            right_pos[:, 2],
            c="black",
            alpha=0.7,
            linewidth=2,
            label="Right EEF Keypose Trajectory",
        )

        # Plot full trajectories with dashed lines in same colors
        if trajectory_positions is not None:
            full_left_pos = trajectory_positions["left"]
            full_right_pos = trajectory_positions["right"]
            ax.plot(
                full_left_pos[:, 0],
                full_left_pos[:, 1],
                full_left_pos[:, 2],
                c="blue",
                alpha=0.5,
                linewidth=1,
                linestyle="--",
                label="Left EEF Full Trajectory",
            )
            ax.plot(
                full_right_pos[:, 0],
                full_right_pos[:, 1],
                full_right_pos[:, 2],
                c="black",
                alpha=0.5,
                linewidth=1,
                linestyle="--",
                label="Right EEF Full Trajectory",
            )

        # Process left hand keyposes
        left_open_mask = ~left_closedness
        left_closed_mask = left_closedness

        if np.any(left_open_mask):
            ax.scatter(
                left_pos[left_open_mask, 0],
                left_pos[left_open_mask, 1],
                left_pos[left_open_mask, 2],
                c="green",
                marker="o",
                s=80,
                label="Open Gripper Keyposes",
                alpha=0.5,
            )

        if np.any(left_closed_mask):
            ax.scatter(
                left_pos[left_closed_mask, 0],
                left_pos[left_closed_mask, 1],
                left_pos[left_closed_mask, 2],
                c="red",
                marker="o",
                s=80,
                label="Closed Gripper Keyposes",
                alpha=0.5,
            )

        # Process right hand keyposes
        right_open_mask = ~right_closedness
        right_closed_mask = right_closedness

        if np.any(right_open_mask):
            ax.scatter(
                right_pos[right_open_mask, 0],
                right_pos[right_open_mask, 1],
                right_pos[right_open_mask, 2],
                c="green",
                marker="o",
                s=80,
                alpha=0.5,
            )

        if np.any(right_closed_mask):
            ax.scatter(
                right_pos[right_closed_mask, 0],
                right_pos[right_closed_mask, 1],
                right_pos[right_closed_mask, 2],
                c="red",
                marker="o",
                s=80,
                alpha=0.5,
            )

    # Set labels and title
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(
        f"Keypose Trajectories - {demo_name}\nEmbodiment: {embodiment_type.value.upper()}\n"
        f"Green: Open Gripper | Red: Closed Gripper\n"
        f"Solid: Keyposes | Dashed: Full Trajectory"
    )
    ax.legend()

    # Make the plot look better
    ax.grid(True, alpha=0.3)

    # Set equal aspect ratio
    max_range = 0.5  # Default range
    if embodiment_type == EmbodimentType.ARM:
        all_pos = positions["arm"]
        if trajectory_positions is not None:
            all_pos = np.vstack([all_pos, trajectory_positions["arm"]])
        ranges = [all_pos[:, i].max() - all_pos[:, i].min() for i in range(3)]
        max_range = max(ranges) / 2
        center = [all_pos[:, i].mean() for i in range(3)]
    else:
        all_pos = np.vstack([positions["left"], positions["right"]])
        if trajectory_positions is not None:
            all_pos = np.vstack(
                [all_pos, trajectory_positions["left"], trajectory_positions["right"]]
            )
        ranges = [all_pos[:, i].max() - all_pos[:, i].min() for i in range(3)]
        max_range = max(ranges) / 2
        center = [all_pos[:, i].mean() for i in range(3)]

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    plt.tight_layout()
    plt.show()


def main():
    """Visualize the keyposes for a given demo."""
    args = ClosedLoopAppArgs().parse_args()
    embodiment = create_embodiment(args.task)

    # Support demo ranges, e.g. 0-9, or a single demo index
    demo_indices = []
    if "-" in args.demos_closed_loop:
        start, end = args.demos_closed_loop.split("-")
        demo_indices = list(range(int(start), int(end) + 1))
    else:
        demo_indices = [int(args.demos_closed_loop)]

    for demo_idx in demo_indices:
        demo_name = f"demo_{str(demo_idx).zfill(5)}"
        print(f"Visualizing {demo_name} with keyposes")
        args.use_keyposes = True
        keypose_policy_states = get_gt_goals(args, demo_name, embodiment, "cuda")
        keypose_positions = extract_keypose_positions(
            keypose_policy_states, embodiment.embodiment_type
        )

        # Get full trajectory positions
        args.use_keyposes = False
        args.gt_goals_subsampling_factor = 1
        trajectory_policy_states = get_gt_goals(args, demo_name, embodiment, "cuda")
        trajectory_positions = extract_keypose_positions(
            trajectory_policy_states, embodiment.embodiment_type
        )

        # Create visualization
        visualize_keyposes_3d(
            keypose_positions, embodiment.embodiment_type, demo_name, trajectory_positions
        )

        print(f"Visualization complete for {demo_name}!")


if __name__ == "__main__":
    main()
