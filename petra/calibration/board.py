from __future__ import annotations

from pathlib import Path
from typing import Annotated

import cv2
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CharucoBoardConfig(BaseModel):
    """Physical ChArUco plate shared by intrinsic calibration and session rectification."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    squares_x: Annotated[int, Field(ge=3)]
    squares_y: Annotated[int, Field(ge=3)]
    square_length_mm: Annotated[float, Field(gt=0)]
    marker_length_mm: Annotated[float, Field(gt=0)]
    dictionary: str = Field(pattern=r"^DICT_[A-Z0-9_]+$")

    @model_validator(mode="after")
    def validate_lengths(self) -> CharucoBoardConfig:
        if self.marker_length_mm >= self.square_length_mm:
            raise ValueError("marker_length_mm must be smaller than square_length_mm")
        return self

    @property
    def chessboard_corner_count(self) -> int:
        """Number of interior chessboard corners the detector can return."""
        return (self.squares_x - 1) * (self.squares_y - 1)

    @property
    def board_size_mm(self) -> tuple[float, float]:
        return (
            self.squares_x * self.square_length_mm,
            self.squares_y * self.square_length_mm,
        )

    @classmethod
    def from_json(cls, path: Path) -> CharucoBoardConfig:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def create_board(self) -> cv2.aruco.CharucoBoard:
        dictionary_id = getattr(cv2.aruco, self.dictionary, None)
        if not isinstance(dictionary_id, int):
            raise ValueError(f"unknown ArUco dictionary: {self.dictionary}")
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        return cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length_mm,
            self.marker_length_mm,
            dictionary,
        )
