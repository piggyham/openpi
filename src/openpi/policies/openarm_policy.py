import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# OpenArm paper-cup relay task instruction (matches the one stored in the
# LeRobot dataset task column, see openarm_mission.dataset.TASK_INSTRUCTION).
TASK_INSTRUCTION = (
    "Use the right arm to move the paper cup from the red area to the "
    "center, then use the left arm to move it to the blue area."
)

# Action dimension of the OpenArm bimanual relay task: left arm Cartesian
# delta (dx, dy, dz, drx, dry, drz) + gripper, then right arm the same.
ACTION_DIM = 14
# State dimension: left 7 joints + left gripper + right 7 joints + right gripper.
STATE_DIM = 16


def make_openarm_example() -> dict:
    """Creates a random input example for the OpenArm policy."""
    return {
        "state": np.random.rand(STATE_DIM),
        "images": {
            "front": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "left_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "right_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        },
        "actions": np.random.rand(10, ACTION_DIM).astype(np.float32),
        "prompt": TASK_INSTRUCTION,
    }


def _parse_image(image) -> np.ndarray:
    """Parse an image to uint8 (H, W, C). LeRobot stores images as float32 (C, H, W)."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class OpenArmInputs(transforms.DataTransformFn):
    """Convert OpenArm observations to the model input format.

    Expects the repacked keys (see ``LeRobotOpenArmDataConfig``): an ``images``
    dict with ``front`` / ``left_wrist`` / ``right_wrist`` RGB views, a 16-dim
    ``state``, 14-dim Cartesian-delta ``actions``, and a ``prompt``. These map
    onto the model's ``base_0_rgb`` / ``left_wrist_0_rgb`` / ``right_wrist_0_rgb``
    image keys. Actions are already deltas, so no extra delta transform is
    needed (same as Libero).
    """

    # Determines which model will be used. Needed for the image mask convention.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["images"]["front"])
        left_wrist_image = _parse_image(data["images"]["left_wrist"])
        right_wrist_image = _parse_image(data["images"]["right_wrist"])

        # Do not change the keys in the dict below.
        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Match the previously validated PI0.5 OpenArm training
                # setup. The right-wrist frames remain in the dataset and can
                # still be replayed, but PI0.5 masks this third image slot.
                "right_wrist_0_rgb": np.False_ if self.model_type == _model.ModelType.PI05 else np.True_,
            },
        }

        # Actions are only available during training. They are padded to the
        # model action_dim (32) by PadStatesAndActions in the model transforms.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class OpenArmOutputs(transforms.DataTransformFn):
    """Convert model outputs back to the OpenArm 14-dim action format."""

    def __call__(self, data: dict) -> dict:
        # Slice out the first ACTION_DIM actions (the rest is padding to the
        # model's action_dim).
        return {"actions": np.asarray(data["actions"][..., :ACTION_DIM])}
