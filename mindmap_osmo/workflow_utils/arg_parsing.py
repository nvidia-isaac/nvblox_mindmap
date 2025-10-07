# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import argparse
from enum import Enum
import sys
from typing import Tuple, Type

from tap import Tap

from mindmap_osmo.workflow_utils.app_arg_overrides import override_app_args
from mindmap_osmo.workflow_utils.workflow_constants import REQUIRED_ARGS, TASK_TYPE_TO_ARG_CLS
from mindmap_osmo.workflow_utils.workflow_types import OsmoTaskType, OsmoWorkflowType


def get_app_args(
    osmo_workflow_type: OsmoWorkflowType,
    osmo_task_type: OsmoTaskType,
    workflow_args: argparse.Namespace,
    input_args: list[str],
) -> Tap:
    """Get the arguments for an osmo task.

    Args:
        osmo_workflow_type: The type of the osmo workflow that runs the application as a osmo task.
        osmo_task_type: The type of the osmo task.
        workflow_args: The workflow arguments.
        input_args: List of arguments to parse

    Returns:
        Tap object containing the parsed application arguments (with overridden defaults).
    """
    # Check if all required arguments are present in the input_args.
    assert args_passed(
        REQUIRED_ARGS[osmo_workflow_type.name], input_args
    ), f"Required arguments not passed on CLI: {REQUIRED_ARGS[osmo_workflow_type.name]}"

    # Parse the application arguments.
    app_args_cls = TASK_TYPE_TO_ARG_CLS[osmo_task_type.name]
    app_args = app_args_cls().parse_args(input_args, known_only=True)

    # Override the application argument defaults with workflow-specific defaults.
    app_args = override_app_args(osmo_workflow_type, osmo_task_type, workflow_args, app_args)

    # Return the application arguments.
    return app_args


def args_passed(required_args: list[str], passed_args: list[str]) -> bool:
    """Check if all required arguments are present in the list of CLI arguments.

    Args:
        required_args (list[str]): The list of required arguments
        passed_args (list[str]): List of arguments to check
    """
    all_args_passed = True
    for arg in required_args:
        if f"--{arg}" not in passed_args:
            all_args_passed = False
            break
    return all_args_passed


def get_non_default_args_str(args: Tap, cls: Type[Tap]) -> str:
    """Get a string containing all non-default arguments from a Tap object.

    This function compares the provided arguments against their default values and
    constructs a command-line style string containing only the arguments that differ
    from defaults.

    Args:
        args (Tap): The Tap object containing argument values to check
        cls (Type[Tap]): The Tap class to instantiate for getting defaults

    Returns:
        str: A string containing all non-default arguments in command-line format.
    """
    assert isinstance(args, cls)
    default_args = cls().parse_args([]).as_dict()
    non_default_args_str = ""
    for key, value in sorted(args.as_dict().items()):
        assert key in default_args, f"Key {key} not in default args"
        if value != default_args[key]:
            non_default_args_str += convert_to_arg_str(key, value)
    return non_default_args_str


def get_log_to_file_arg_string(logfile_name: str) -> str:
    return " \\\n" + f"     2>&1 | tee {{{{output}}}}/{logfile_name}\n"


def convert_to_arg_str(arg_name, value):
    """Convert a value to a string that can be used as an argument.

    Args:
        arg_name (str): The name of the argument
        value: The value to convert

    Returns:
        str: A string containing the argument in command-line format.
    """
    if isinstance(value, bool):
        # boolean args are written as flags
        if value:
            return f" \\\n    --{arg_name}"
    elif isinstance(value, Enum):
        return f" \\\n    --{arg_name} {value.value}"
    elif isinstance(value, list):
        if value and isinstance(value[0], list):
            # Handle list of lists by joining inner lists with spaces and outer with commas
            list_of_lists_str = '"' + str(value) + '"'
            return f" \\\n    --{arg_name} {list_of_lists_str}"
        else:
            list_str = " ".join(str(x) for x in value)
            return f" \\\n    --{arg_name} {list_str}"
    elif isinstance(value, tuple):
        tuple_str = " ".join(str(x) for x in value)
        return f" \\\n    --{arg_name} {tuple_str}"
    else:
        return f" \\\n    --{arg_name} {value}"
