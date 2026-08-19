# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Camera data for OpenArm Dataset."""

import os
from pathlib import Path
from collections.abc import Iterator

import numpy as np
from PIL import Image


class Frame:
    """An image in camera."""

    def __init__(self, path: os.PathLike):
        """Initialize Frame at the path."""
        self.path = path
        self.timestamp: float = self._get_timestamp()

    def __eq__(self, other):
        """Compare whether the other is the same path Frame or not."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.path == other.path

    def load(self) -> np.ndarray:
        """Load image of this frame.

        Returns:
            Image array.

        """
        return np.array(Image.open(self.path))

    def show(self):
        """Show image of this frame."""
        return Image.open(self.path).show()

    def _get_timestamp(self) -> float:
        return float(Path(self.path).stem) / 1e9


class Camera:
    """Camera for OpenArm Dataset."""

    def __init__(
        self,
        name: str,
        base_path: str | os.PathLike,
    ):
        """Initialize Camera."""
        self.name: name = name
        self.base_path: os.PathLike = Path(base_path)
        self.all_files: list[Path] = (
            sorted(f for f in self.base_path.iterdir() if f.is_file())
            if self.base_path.exists()
            else []
        )

    @property
    def num_frames(self) -> int:
        """Get number of frames."""
        return len(self.all_files)

    def get_frame(self, index: int) -> Frame:
        """Get frame at the index.

        Args:
            index: Index to get.

        Returns:
            Frame at the index.

        """
        return Frame(self.all_files[index])

    def frames(self) -> Iterator[Frame]:
        """Iterate all frames.

        Returns:
            Iterator of Frame.

        """
        for file in self.all_files:
            yield Frame(file)

    def load_timestamps(self) -> list[float]:
        """Load timestamps.

        Returns:
            List of Unix time.

        """
        return [frame.timestamp for frame in self.frames()]
