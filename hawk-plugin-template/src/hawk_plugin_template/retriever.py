#
# Example Hawk retriever
#

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterator

from PIL import Image

from hawk.detection import Detection
from hawk.hawkobject import HawkObject
from hawk.objectid import ObjectId
from hawk.scout.retrieval.retriever import Retriever, RetrieverConfig


class ExampleRetrieverConfig(RetrieverConfig):
    index_path: Path
    timeout: float = 0.1  # override the default timeout set by RetrieverConfig


class ExampleRetriever(Retriever):
    config_class = ExampleRetrieverConfig
    config: ExampleRetrieverConfig

    def _validate_path(self, path: Path) -> None:
        # Path.relative_to raises ValueError when it is not a subpath.
        path.resolve().relative_to(self.config.data_root)

    def get_next_objectid(self) -> Iterator[ObjectId | None]:
        # Make sure the provided path is within the data root.
        self._validate_path(self.config.index_path)

        # (in the current implementation the index is a list of filenames, but
        # here we could be extracting keys from a SQL table, arguments needed
        # to clip fragments from a larger image or video source, etc.)
        for line in self.config.index_path.open():
            # yield an object id for the current file
            yield ObjectId(line.strip())

            # send a sentinel value to indicate that the retriever has
            # completed tiling an image. This will be used to increment the
            # 'images retrieved' statistic and introduce an optional per-object
            # delay, specified as the 'timeout' parameter in the retriever
            # configuration.
            yield None

    def get_ml_data(self, objectid: ObjectId) -> HawkObject:
        object_path = Path(objectid.oid)
        self._validate_path(object_path)

        # Have HawkObject.from_file read the file data and guess the media type
        # based on the filename extension.
        return HawkObject.from_file(object_path)

    def get_oracle_data(self, objectid: ObjectId) -> list[HawkObject]:
        object_path = Path(objectid.oid)
        self._validate_path(object_path)

        # convert the input to a thumbnail image to be displayed by Hawk GUI
        # in a web browser.
        with Image.open(object_path) as img:
            img.thumbnail((256, 256))
            with BytesIO() as buffer:
                img.save(buffer, format="PNG")
                content = buffer.getvalue()
        return [HawkObject(content=content, media_type="image/png")]

    def get_groundtruth(self, objectid: ObjectId) -> list[Detection]:
        # if we have no known detections or classifications results
        # we can return an empty list here.
        return []
