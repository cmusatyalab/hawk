# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

"""Hawk Object datatype. Wraps binary content with an associated media type."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .proto.messages_pb2 import _HawkObject as pb2_HawkObject

# is there a better way to do this?
# media type -> filename suffix mapping
MEDIA_TYPES = {
    "audio/mpeg": [".mp3"],
    "audio/snd": [".snd"],
    "audio/wav": [".wav"],
    "binary/octet-stream": [".bin"],
    "image/avif": [".avif"],
    "image/gif": [".gif"],
    "image/jpeg": [".jpeg", ".jpg", ".jfif", ".pjpeg", ".pjp"],
    "video/mp4": [".mp4"],
    "image/png": [".png", ".apng"],
    "image/pnm": [".pnm", ".ppm", ".pgm", ".pbm"],
    "image/svg+xml": [".svg"],
    "image/tiff": [".tiff", ".tif", ".geotiff"],
    "image/webp": [".webp"],
    "x-array/numpy": [".npy"],
    "x-array/numpyz": [".npz"],
    "x-tensor/pytorch": [".pt"],
    "x-tensor/tensorflow": [".keras"],
}
MEDIA_SUFFIXES = {
    suffix: media_type
    for media_type, suffixes in MEDIA_TYPES.items()
    for suffix in suffixes
}


@dataclass
class HawkObject:
    content: bytes
    media_type: str

    @classmethod
    def from_file(cls, object_path: Path) -> HawkObject:
        """Identify media type based on file suffix and read object content from file.

        raises FileNotFoundError if the media type is unknown or object cannot be read.
        """
        try:
            media_type = MEDIA_SUFFIXES[object_path.suffix]
        except KeyError as e:
            msg = f"Unknown media type for {object_path.name}"
            raise FileNotFoundError(msg) from e

        content = object_path.read_bytes()
        return cls(content, media_type)

    def to_file(self, path: Path) -> Path:
        """Writes the object to a file. Returns path to the written file."""
        object_path = path.with_suffix(self.suffix)
        object_path.write_bytes(self.content)
        return object_path

    @classmethod
    def from_protobuf(cls, msg: bytes | pb2_HawkObject) -> HawkObject:
        """Parses a HawkObject from a protobuf message."""
        if isinstance(msg, bytes):
            obj = pb2_HawkObject()
            obj.ParseFromString(msg)
        else:
            obj = msg
        if obj.media_type not in MEDIA_TYPES:
            raise ValueError(f"Unknown media type: {obj.media_type}")
        return cls(obj.content, obj.media_type)

    def to_protobuf(self) -> pb2_HawkObject:
        """Returns a protobuf representation of the HawkObject."""
        return pb2_HawkObject(
            content=self.content,
            media_type=self.media_type,
        )

    @property
    def suffix(self) -> str:
        """Return file suffix."""
        return MEDIA_TYPES[self.media_type][0]
