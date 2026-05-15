"""MongoDB Atlas vector index manager."""

from __future__ import annotations

import logging

from pymongo.collection import Collection
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)


def ensure_vector_index(
    collection: Collection,
    index_name: str,
    dimensions: int,
) -> None:
    definition = {
        "name": index_name,
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": dimensions,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "document_id"},
                {"type": "filter", "path": "metadata.source"},
            ]
        },
    }
    try:
        collection.database.command(
            {
                "createSearchIndexes": collection.name,
                "indexes": [definition],
            }
        )
    except PyMongoError as exc:
        # Most clusters will return an error when index already exists.
        logger.info("Vector index creation skipped: %s", exc)
