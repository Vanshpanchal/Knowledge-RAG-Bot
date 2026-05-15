"""Database connection management and lifecycle."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from typing import Any

from fastapi import FastAPI
from pymongo import MongoClient
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError
from pymongo.server_api import ServerApi

from app.core.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Connect to MongoDB on startup, close it on shutdown."""
    mongo_client: MongoClient | None = None
    try:
        mongo_client = MongoClient(
            settings.MONGODB_URI,
            server_api=ServerApi("1"),
            serverSelectionTimeoutMS=7000,
            connectTimeoutMS=7000,
            socketTimeoutMS=15000,
        )
        mongo_client.admin.command("ping")
        app.state.mongo_client = mongo_client
        app.state.mongo_db = mongo_client[settings.DATABASE_NAME]
        logger.info("MongoDB connected to %s", settings.DATABASE_NAME)
    except ServerSelectionTimeoutError as exc:
        logger.exception("MongoDB unavailable: %s", exc)
        app.state.mongo_client = None
        app.state.mongo_db = None
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("MongoDB connection failed: %s", exc)
        app.state.mongo_client = None
        app.state.mongo_db = None

    yield

    if mongo_client is not None:
        mongo_client.close()
        logger.info("MongoDB connection closed")


def get_mongo_client(app: FastAPI) -> MongoClient | None:
    return getattr(app.state, "mongo_client", None)


def get_mongo_db(app: FastAPI):
    return getattr(app.state, "mongo_db", None)


def ensure_indexes(app: FastAPI) -> None:
    """Ensure basic indexes required by ingestion and retrieval."""
    db = get_mongo_db(app)
    if db is None:
        return

    try:
        db[settings.DOCUMENT_COLLECTION].create_index("document_id", unique=True)
        db[settings.DOCUMENT_COLLECTION].create_index("status")
        db[settings.CHUNK_COLLECTION].create_index("chunk_id", unique=True)
        db[settings.CHUNK_COLLECTION].create_index([("document_id", 1), ("chunk_index", 1)])
        db[settings.CHUNK_COLLECTION].create_index("text")
    except PyMongoError as exc:
        logger.warning("Could not create MongoDB indexes: %s", exc)
