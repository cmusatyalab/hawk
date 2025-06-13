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

from .classes import NEGATIVE_CLASS, ClassName
from .rusty import unwrap, unwrap_or

# Namespace identifier for UUIDs derived from Hawk ObjectIDs
HAWK_OBJECTID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "hawk.elijah.cs.cmu.edu")

# Regex to match a Hawk ObjectID
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
        # but a cached property makes initialization and inheritence easier.
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

    # compat functions
    def file_name(
        self, parent: Path | None = None, file_ext: str | None = None
    ) -> Path:
        if file_ext is None:
            file_ext = unwrap(self._file_path()).suffix

        path = Path(str(self.shortid)).with_suffix(file_ext)
        return parent.joinpath(path) if parent is not None else path

    def _groundtruth(self) -> ClassName:
        """Extract groundtruth from an objectid.

        This is somewhat compatible with how this information was extracted
        from the old string based objectids, so it is useful for backward
        compatibility, but OTOH it only works properly for a subset of
        retrievers (f.i. it breaks for video_retriever).

        Also this information should not be encoded in the objectid itself,
        extracing this information should be a retriever specific function.

        We must always avoid referencing the ground truth data and make sure it
        is only used for debugging and mission evaluation purposes only.
        """
        m = re.match(OID_RE, self.oid)
        if m is None:
            return NEGATIVE_CLASS
        return ClassName(sys.intern(m.group("gt")))

    def _file_path(self, data_root: Path | None = None) -> Path | None:
        """Extract a file path/name from an objectid.

        This is somewhat compatible with how this information was extracted
        from the old string based objectids, so it is useful for backward
        compatibility, but OTOH it only works properly for a subset of
        retrievers (f.i. it breaks for video_retriever).

        Not convinced that anyone, aside from the retrievers, should really
        know or have access to this information to begin with.
        """
        m = re.match(OID_RE, self.oid)
        if m is None:
            return None

        path = m.group("path")
        if data_root is None:
            return Path(path)

        full_path = data_root.joinpath(path)
        if data_root not in full_path.parents:
            return None

        return full_path


@dataclass(frozen=True)
class ExampleObjectId(ObjectId):
    """Example 'retriever specific' objectid class.

    This would be used internally by a retriever if it needs to, for some
    reason, interpret the content of the object id value.
    """

    DATA_DIR: Path = Path("/datadir")

    @classmethod
    def from_objectid(cls, oid: ObjectId) -> ExampleObjectId:
        """Validate the object id is valid in the context of this retriever."""
        raw_oid = oid.serialize_oid()
        if not raw_oid.startswith("example:"):
            # KeyError because objectid is badly formatted.
            # Maybe this could/should be part of the object-specific constructor.
            msg = "Invalid ObjectID for ExampleRetriever"
            raise KeyError(msg)

        # we've validated the oid is the right format, so we can instantiate
        specific_oid = cls(raw_oid)

        path = specific_oid._file_path()
        if cls.DATA_DIR not in path.parents or not path.exists():
            # ValueError because object referenced by objectid does not exist.
            # This would only only be relevant when we get a request for the
            # ML-ready or Oracle-ready data.
            msg = "Invalid ObjectID for ExampleRetriever"
            raise ValueError(msg)

        return specific_oid

    def _file_path(self, data_root: Path | None = None) -> Path:
        """Return the path to the on-disk copy of the object.

        This assumes our object ids are `example:path/in/data_dir/object.ext`
        """
        data_root = unwrap_or(data_root, self.DATA_DIR)
        return data_root.joinpath(self.oid[8:]).resolve()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Hawk object-id to short-id to locate related resources"
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
