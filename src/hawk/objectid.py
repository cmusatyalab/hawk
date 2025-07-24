# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

"""Hawk ObjectId datatype."""

from __future__ import annotations

import re
import sys
import uuid
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from .classes import ClassName
from .proto import common_pb2

# Namespace identifier for UUIDs derived from Hawk ObjectIDs
HAWK_OBJECTID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "hawk.elijah.cs.cmu.edu")

# Regex to match a Hawk Legacy ObjectID
OID_RE = r"^/(?P<gt>.*)/collection/id/(?P<path>.*)$"


@dataclass(frozen=True)
class ObjectId:
    """Class representing a unique Hawk Object identifier.

    Because an ObjectId could contain any characters and are for all intents
    and purposes not restricted in size, we use UUIDs derived from the value of
    the object id (shortid) as cache lookup keys, to identify duplicate items,
    and to create unique on-disk file names,
    """

    oid: str

    @cached_property
    def shortid(self) -> uuid.UUID:
        """Return (cached) UUID representing the ObjectId's value."""
        # we likely use this often enough to always compute it in __post_init__
        # but a cached property makes initialization and inheritance easier.
        return uuid.uuid5(HAWK_OBJECTID_NAMESPACE, self.oid)

    def __eq__(self, other: object) -> bool:
        """Compare against UUID (shortid) or other ObjectId objects."""
        if isinstance(other, uuid.UUID):
            return self.shortid == other
        return isinstance(other, ObjectId) and self.shortid == other.shortid

    def __hash__(self) -> int:
        """Rely on the fact that the shortid is just as unique as the objectid."""
        return self.shortid.__hash__()

    def __repr__(self) -> str:
        return f"ObjectId('{self.oid}')"

    def serialize_oid(self) -> str:
        """Get the objectid value so we can store it or send across a network.

        Using this as the accessor should make it easy to find all places we do
        something special with the ObjectId (git grep serialize_oid).
        """
        return self.oid

    @classmethod
    def from_protobuf(cls, msg: bytes | common_pb2.ObjectId) -> ObjectId:
        """Parses an ObjectId from a protobuf message."""
        if isinstance(msg, bytes):
            obj = common_pb2.ObjectId()
            obj.ParseFromString(msg)
        else:
            obj = msg
        return cls(obj.oid)

    def to_protobuf(self) -> common_pb2.ObjectId:
        """Returns a protobuf representation of the ObjectId."""
        return common_pb2.ObjectId(oid=self.oid)

    def file_name(self, parent: Path | None = None, file_ext: str = ".bin") -> Path:
        path = Path(str(self.shortid)).with_suffix(file_ext)
        return parent.joinpath(path) if parent is not None else path


@dataclass(frozen=True)
class LegacyObjectId(ObjectId):
    object_path: Path
    groundtruth: ClassName

    @classmethod
    def from_objectid(cls, object_id: ObjectId) -> LegacyObjectId:
        oid = object_id.serialize_oid()
        m = re.match(OID_RE, oid)
        assert m is not None, "unable to parse object id"

        object_path = Path(m.group("path"))
        groundtruth = ClassName(sys.intern(m.group("gt")))
        return cls(oid=oid, object_path=object_path, groundtruth=groundtruth)

    def file_path(self, data_root: Path | None = None) -> Path | None:
        """Extract a file path/name from an objectid.

        This is somewhat compatible with how this information was extracted
        from the old string based objectids, so it is useful for backward
        compatibility, but OTOH it only works properly for a subset of
        retrievers (f.i. it breaks for video_retriever).

        Not convinced that anyone, aside from the retrievers, should really
        know or have access to this information to begin with.
        """
        if data_root is None:
            return self.object_path

        full_path = data_root.joinpath(self.object_path).resolve()
        if data_root not in full_path.parents:
            return None

        return full_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Hawk object-id to short-id to locate related resources",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display both short-id and object-id",
    )
    parser.add_argument("oid", type=ObjectId, nargs="*", help="Object ID to parse")
    args = parser.parse_args()

    oids = [ObjectId(line.strip()) for line in sys.stdin] if not args.oid else args.oid

    for oid in oids:
        if args.verbose:
            print(oid.shortid, oid.serialize_oid())
        else:
            print(oid.shortid)
