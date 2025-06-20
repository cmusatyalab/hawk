# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

"""Hawk Object datatype. Wraps binary content with an associated media type.

Provides functions to convert to/from protobuf and to/from a file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .proto import common_pb2

# Is there a better way to do this?
# This breaks down for audio/video formats because there are several containers
# that can contain a wide range of different formats, an mp4 can be audio-only,
# video-only, or both, or even the same data encoded with different codecs.

# Simple media type -> filename suffix map
MEDIA_TYPES = {
    "application/json": [".json"],
    "application/xml": [".xml"],
    "application/yaml": [".yaml", ".yml"],
    "audio/aac": [".aac"],
    "audio/flac": [".flac"],
    "audio/mpeg": [".mp3"],
    "audio/snd": [".snd"],
    "audio/wav": [".wav"],
    "binary/octet-stream": [".bin"],
    "image/apng": [".apng"],
    "image/avif": [".avif"],
    "image/bmp": [".bmp"],
    "image/gif": [".gif"],
    "image/ico": [".ico", ".cur"],
    "image/jpeg": [".jpg", ".jpeg", ".jpe", ".jfif", ".pjpeg", ".pjp"],
    "image/png": [".png"],
    "image/pnm": [".pnm", ".ppm", ".pgm", ".pbm"],
    "image/svg+xml": [".svg"],
    "image/tiff": [".tif", ".tiff"],
    "image/webp": [".webp"],
    "image/xbm": [".xbm"],
    "text/plain": [".txt"],
    "video/mp4": [".mp4"],
    "video/ogg": [".ogg"],
    "x-array/numpy": [".npy"],
    "x-array/numpyz": [".npz"],
    "x-tensor/pytorch": [".pt"],
    "x-tensor/tensorflow": [".keras"],
}
# And here we construct the reverse filename suffix -> media type map
MEDIA_SUFFIXES = {
    suffix: media_type
    for media_type, suffixes in MEDIA_TYPES.items()
    for suffix in suffixes
}


class MediaTypeError(FileNotFoundError):
    """Raised when a media type is unknown."""


@dataclass
class HawkObject:
    content: bytes
    media_type: str

    ## Importers

    @classmethod
    def from_file(cls, object_path: Path) -> HawkObject:
        """Identify media type based on file suffix and read object content from file.

        raises MediaTypeError if the media type is unknown or FileNotFoundError
        when the object cannot be read.

        MediaTypeError is a subclass of FileNotFoundError so that both can be
        caught by a single except clause.
        """
        try:
            media_type = MEDIA_SUFFIXES[object_path.suffix]
        except KeyError as e:
            msg = f"Unknown media type for {object_path.name}"
            raise MediaTypeError(msg) from e

        content = object_path.read_bytes()
        return cls(content, media_type)

    @classmethod
    def from_protobuf(cls, msg: bytes | common_pb2.HawkObject) -> HawkObject:
        """Parses a HawkObject from a protobuf message.

        raises MediaTypeError if the media type is unknown.
        """
        if isinstance(msg, bytes):
            obj = common_pb2.HawkObject()
            obj.ParseFromString(msg)
        else:
            obj = msg

        if obj.media_type not in MEDIA_TYPES:
            raise MediaTypeError(f"Unknown media type: {obj.media_type}")
        return cls(obj.content or b"", obj.media_type or "binary/octet-stream")

    ## Helpers

    @property
    def suffix(self) -> str:
        """Return file suffix for the object based on it's media type."""
        return MEDIA_TYPES[self.media_type][0]

    def file_path(self, path: Path, index: int | None = None) -> Path:
        """Update a path to the object file to use a proper file-type based
        suffix and optionally a sequence number in case there are multiple
        derived artifacts."""
        if not index:
            return path.with_suffix(self.suffix)
        return path.with_suffix(f".{index}.{self.suffix}")

    ## Exporters

    def to_file(
        self, path: Path, index: int | None = None, *, mkdirs: bool = False
    ) -> Path:
        """Write the object to a file. Return path to the new file.

        index is used when we need to write multiple file for the same objectid.
        mkdirs indicates if we need to create any missing parent directories.
        """
        object_path = self.file_path(path, index)
        if mkdirs:
            object_path.parent.mkdir(parents=True, exist_ok=True)
        object_path.write_bytes(self.content)
        return object_path

    def to_protobuf(self) -> common_pb2.HawkObject:
        """Returns a protobuf representation of the HawkObject."""
        return common_pb2.HawkObject(
            content=self.content,
            media_type=self.media_type,
        )
