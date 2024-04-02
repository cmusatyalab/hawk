# SPDX-FileCopyrightText: 2022-2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only

"""Utility functions for labelers"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import pandas as pd

if TYPE_CHECKING:
    import os

_UNLABELED_DTYPES = {
    "index": "int",
    "objectId": "string",
    "scoutIndex": "int",
    "size": "int",
    "score": "float",
}

_LABELED_DTYPES = {
    "index": "int",
    "imageLabel": "Int8",
    # What we probably want...
    # "boundingBoxes": "list[tuple[float,float,float,float]]",
    # What home/scout currently expect...
    # "boundingBoxes": "list[str]",
    # What pandas can actually recognize...
    "boundingBoxes": "object",
}


@dataclass
class Result:
    index: int
    objectId: str
    scoutIndex: int
    score: float
    size: int
    data: bytes | None = None
    imageLabel: int | None = None
    boundingBoxes: list[tuple[float, float, float, float]] | None = None

    def to_unlabeled_json(self, **kwargs: int | float | str) -> str:
        return json.dumps(
            dict(
                index=self.index,
                objectId=self.objectId,
                scoutIndex=self.scoutIndex,
                score=self.score,
                size=self.size,
                **kwargs,
            ),
            sort_keys=True,
        )

    def to_labeled_jsonl(self, jsonl_file: os.PathLike[str] | str) -> None:
        json_data = json.dumps(
            dict(
                index=self.index,
                objectId=self.objectId,
                scoutIndex=self.scoutIndex,
                size=self.size,
                imageLabel=self.imageLabel,
                boundingBoxes=self.boundingBoxes,
            )
        )
        with Path(jsonl_file).open("a") as fd:
            fd.write(f"{json_data}\n")

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Iterator[Result]:
        for item in df.dropna().itertuples():
            yield cls(**item._asdict())

    @staticmethod
    def to_dataframe(results: list[Result] | None = None) -> pd.DataFrame:
        """returns a dataframe with all the columns/datatypes defined"""
        # poorly documented dictionary merge was introduced in Python 3.5
        merged_dtypes = {**_UNLABELED_DTYPES, **_LABELED_DTYPES}
        if results is None:
            results = []
        return (
            pd.DataFrame(results, columns=list(merged_dtypes))
            .astype(merged_dtypes)
            .set_index("index")
        )

    # @classmethod
    # def from_sendtiles_msg(cls, msg: bytes) -> Result:
    #    request = SendTiles()
    #    request.ParseFromString(msg)
    #    return cls(
    #        objectId=request.objectId,
    #        scoutIndex=request.scoutIndex,
    #        score=request.score,
    #        size=request.ByteSize(),
    #        data=request.attributes["thumbnail.jpeg"],
    #    )


@dataclass
class MissionResults:
    mission_dir: os.PathLike[str] | str = field(default_factory=Path.cwd)
    sync_labels: bool = False

    _unlabeled_offset: int = 0
    _labeled_offset: int = 0

    unlabeled_jsonl: Path = field(init=False, repr=False)
    labeled_jsonl: Path = field(init=False, repr=False)
    df: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        self.unlabeled_jsonl = Path(self.mission_dir, "unlabeled.jsonl")
        self.labeled_jsonl = Path(self.mission_dir, "labeled.jsonl")
        self.df = self.empty_dataframe

    def __iter__(self) -> MissionResults:
        self.rows_seen = self.df.index.size
        return self

    def __next__(self) -> pd.DataFrame:
        while not self.resync_unlabeled():
            time.sleep(0.5)
        if self.sync_labels:
            self.resync_labeled()
        prev_offset = self.rows_seen
        self.rows_seen = self.df.index.size
        return self.df[prev_offset:]

    def _read_jsonl_tail(
        self, jsonl_file: Path, offset: int, dtypes: dict[str, str]
    ) -> tuple[pd.DataFrame | None, int]:
        """Read entries from a jsonl file starting at offset.
        Returns:
        - None when there was nothing new to read.
        - An empty dataframe and offset 0 when the file was empty.
        - dataframe with read entries and the file length (new offset).
        """
        length = jsonl_file.stat().st_size if jsonl_file.exists() else 0

        # no changes
        if offset == length:
            return None, offset

        # length (re)set to 0, return empty dataframe
        if length == 0:
            return self.empty_dataframe, 0

        with jsonl_file.open() as fp:
            # seek to previous tail
            fp.seek(offset)

            # read new data, forcing the data types as we go
            tail = (
                pd.DataFrame(
                    pd.read_json(fp, orient="records", lines=True, dtype=dtypes),
                    columns=list(dtypes),
                )
                .astype(dtypes)
                .set_index("index")
            )
            # record new tail
            offset = fp.tell()
        return tail, offset

    def resync(self) -> bool:
        updated = self.resync_unlabeled()
        if self.sync_labels:
            updated |= self.resync_labeled()
        return updated

    def resync_unlabeled(self) -> bool:
        new_data, self._unlabeled_offset = self._read_jsonl_tail(
            self.unlabeled_jsonl,
            self._unlabeled_offset,
            dtypes=_UNLABELED_DTYPES,
        )
        if new_data is None:
            return False

        # concatenate new tail
        self.df = pd.concat([self.df, new_data]) if self._unlabeled_offset else new_data
        return True

    def resync_labeled(self) -> bool:
        new_data, self._labeled_offset = self._read_jsonl_tail(
            self.labeled_jsonl,
            self._labeled_offset,
            dtypes=_LABELED_DTYPES,
        )
        if new_data is None:
            return False

        # merge labels
        self.df[["imageLabel", "boundingBoxes"]] = (
            self.df[["imageLabel", "boundingBoxes"]].combine_first(new_data)
            if self._labeled_offset
            else pd.NA
        )
        return True

    def reset_labels(self) -> None:
        self.df["imageLabel", "boundingBoxes"] = pd.NA
        self._labeled_offset = 0

    @property
    def unlabeled(self) -> pd.Series[bool]:
        """Using this so much, figured a helper could be useful."""
        return self.df.imageLabel.isna()

    def save_new_labels(self, labels: pd.Series[int]) -> None:
        # make sure we're current with the on-disk state and capture the
        # previous unlabeled state before we merge the new labels.
        self.resync_unlabeled()
        self.resync_labeled()
        unlabeled = self.df.imageLabel.isna()

        # drop new labels with negative or NA/NaN/None values and merge
        updates = labels[labels >= 0].astype("Int8")
        self.df["imageLabel"] = self.df.imageLabel.combine_first(updates)

        # and filter down to the list of updated labels
        new_labels = self.df[unlabeled & self.df.imageLabel.notna()]
        if new_labels.empty:
            return

        # ugly workaround for not being able to do .fillna([])
        # new_labels["boundingBoxes"] = new_labels.boundingBoxes.fillna("").apply(list)

        with self.labeled_jsonl.open("a") as fp:
            new_labels.reset_index()[
                [
                    "index",
                    "objectId",
                    "scoutIndex",
                    "size",
                    "imageLabel",
                    "boundingBoxes",
                ]
            ].to_json(fp, orient="records", lines=True)

    @property
    def empty_dataframe(self) -> pd.DataFrame:
        """returns an empty dataframe with all the columns/datatypes defined"""
        return Result.to_dataframe()
