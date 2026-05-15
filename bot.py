#!/usr/bin/env python3
"""Telegram bot integrating RAG audio/text upload, query, and TTS reply."""

from __future__ import annotations

import io
import re
import json
import logging
import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    Update,
    InputFile,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Read config from environment
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
# Use public service URL on Render (RENDER_EXTERNAL_URL), or custom API URL, or localhost fallback
API_BASE_URL = os.getenv(
    "CORTEX_BASE_URL",  # Custom override
    os.getenv(
        "RENDER_EXTERNAL_URL",  # Render provides this automatically
        f"http://127.0.0.1:{os.getenv('PORT', '8000')}",  # Local fallback
    ),
).rstrip("/")
RAG_API_URL = f"{API_BASE_URL}/api/v1"
SERVER_CHECK_URL = "https://cortex-engine-latest.onrender.com"
RAG_API_KEY = os.getenv("RAG_API_KEY", "")
APPWRITE_API_KEY = os.getenv("APPWRITE_API_KEY", "")
APPWRITE_PROJECT_ID = os.getenv("APPWRITE_PROJECT_ID", "")
APPWRITE_ENDPOINT = os.getenv("APPWRITE_ENDPOINT", "")
MIME_EXTENSION_MAP = {
    "application/pdf": ".pdf",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/ogg": ".ogg",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "application/json": ".json",
    "application/zip": ".zip",
}
ALLOWED_USERS = {
    int(user_id.strip())
    for user_id in os.getenv("ALLOWED_USERS", "834727332").split(",")
    if user_id.strip().isdigit()
}

if not RAG_API_KEY:
    logger.warning(
        "RAG_API_KEY is empty; requests to the API will fail with 401 Unauthorized"
    )


MAIN_KEYBOARD = ReplyKeyboardMarkup(
    [
        [KeyboardButton("Start"), KeyboardButton("Help")],
        [KeyboardButton("Query"), KeyboardButton("Query Doc")],
        [KeyboardButton("Text"), KeyboardButton("Stats")],
        [KeyboardButton("Reset"), KeyboardButton("ID")],
    ],
    resize_keyboard=True,
    one_time_keyboard=False,
)

QUERY_MODE_KEY = "pending_query_mode"

ACTIONS_INLINE_KEYBOARD = InlineKeyboardMarkup(
    [
        [InlineKeyboardButton(text="Start", switch_inline_query_current_chat="/start")],
        [
            InlineKeyboardButton(text="Server", callback_data="server_status"),
        ],
    ]
)


def is_allowed_user(update: Update) -> bool:
    """Check numeric Telegram user ID against the allowlist."""
    user = update.effective_user
    if user is None:
        return False
    if not ALLOWED_USERS:
        return False
    return user.id in ALLOWED_USERS


async def reject_unauthorized(update: Update) -> None:
    """Reject unauthorized users without leaking bot internals."""
    if update.message:
        await update.message.reply_text("Unauthorized")


async def require_authorized(update: Update) -> bool:
    """Return False and notify the user if they are not in the allowlist."""
    if is_allowed_user(update):
        return True
    await reject_unauthorized(update)
    logger.warning(
        "Blocked Telegram user %s", getattr(update.effective_user, "id", None)
    )
    return False


def api_post_text(endpoint: str, payload: dict) -> dict:
    """POST JSON to RAG API."""
    url = f"{RAG_API_URL}/{endpoint}"
    headers = {
        "Authorization": f"Bearer {RAG_API_KEY}" if RAG_API_KEY else "",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.exception(f"API POST failed for {endpoint}: {exc}")
        raise


def api_post_file(
    endpoint: str,
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    extra_data: dict | None = None,
) -> dict:
    """POST multipart file to RAG API."""
    url = f"{RAG_API_URL}/{endpoint}"
    headers = {
        "Authorization": f"Bearer {RAG_API_KEY}" if RAG_API_KEY else "",
    }

    files = {"file": (filename, io.BytesIO(file_bytes), mime_type)}
    data = extra_data or {}

    try:
        resp = requests.post(url, files=files, data=data, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.exception(f"API POST file failed for {endpoint}: {exc}")
        raise


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message."""
    if not await require_authorized(update):
        return

    welcome = (
        "🎤 **Knowledge RAG Audio Bot**\n\n"
        "Buttons:\n"
        "`Start` - This help message\n"
        "`Help` - Show this help\n"
        "`Query` - Ask a question\n"
        "`Query Doc` - Ask a question with cited material and a download\n"
        "`Text` - Save a text note\n"
        "`Stats` - Show bot stats\n"
        "`Reset` - Reset the conversation\n"
        "`ID` - Show your Telegram numeric ID\n\n"
        "Slash commands are still supported too:\n"
        "`/start`, `/help`, `/query`, `/query_doc`, `/text`, `/stats`, `/reset`, `/id`\n\n"
        "📎 Send a file (PDF/image/audio) to upload\n"
        "🎵 Send a voice message to ask a question and get audio reply\n"
        "🔊 Send an audio file to query the knowledge base\n"
    )
    await update.message.reply_text(
        welcome,
        parse_mode="Markdown",
        reply_markup=ACTIONS_INLINE_KEYBOARD,
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the command list."""
    if not await require_authorized(update):
        return

    await start(update, context)


async def query_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Prompt the user for a normal query."""
    if not await require_authorized(update):
        return

    context.user_data[QUERY_MODE_KEY] = "query"
    await update.message.reply_text(
        "Send your question and I’ll answer it.", reply_markup=MAIN_KEYBOARD
    )


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Alias for /query command."""
    await cmd_query(update, context)


async def query_doc_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Prompt the user for a query with citations and a downloadable summary."""
    if not await require_authorized(update):
        return

    question = " ".join(context.args) if context.args else ""
    if question:
        try:
            await update.message.reply_text("🔎 Searching with citations...")
            await _send_query_response(update, question, include_citations=True)
        except Exception as exc:
            logger.exception("Cited query failed")
            await update.message.reply_text(f"❌ Error: {exc}")
        return

    context.user_data[QUERY_MODE_KEY] = "query_doc"
    await update.message.reply_text(
        "Send your question and I’ll return the answer, citations, and a downloadable summary.",
        reply_markup=MAIN_KEYBOARD,
    )


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show bot stats placeholder."""
    if not await require_authorized(update):
        return

    await update.message.reply_text(
        "Stats are not implemented yet.", reply_markup=MAIN_KEYBOARD
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reset the conversation state placeholder."""
    if not await require_authorized(update):
        return

    await update.message.reply_text(
        "Conversation reset. Send a new question or file.",
        reply_markup=MAIN_KEYBOARD,
    )


async def id_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the numeric Telegram user ID."""
    if not await require_authorized(update):
        return

    user = update.effective_user
    user_id = getattr(user, "id", None)
    await update.message.reply_text(
        f"Your Telegram user ID is: `{user_id}`",
        parse_mode="Markdown",
        reply_markup=MAIN_KEYBOARD,
    )


async def server_status_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Check hosted server endpoint and report status."""
    query = update.callback_query
    if query is None:
        return

    if not is_allowed_user(update):
        await query.answer("Unauthorized", show_alert=True)
        return

    await query.answer("Checking server...")
    started = time.monotonic()
    try:
        response = requests.get(SERVER_CHECK_URL, timeout=20)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        await query.message.reply_text(
            f"Server check: {response.status_code} in {elapsed_ms} ms\n{SERVER_CHECK_URL}"
        )
    except Exception as exc:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        await query.message.reply_text(
            f"Server check failed after {elapsed_ms} ms\n{SERVER_CHECK_URL}\nError: {exc}"
        )


def _format_citations(citations: list[dict]) -> str:
    if not citations:
        return "No citations were returned."

    lines: list[str] = []
    for index, citation in enumerate(citations[:5], start=1):
        source = citation.get("source") or "Unknown source"
        page = citation.get("page")
        chunk_id = citation.get("chunk_id") or "N/A"
        score = citation.get("score")
        text = str(citation.get("text") or "").strip()
        excerpt = text[:300] + ("..." if len(text) > 300 else "")
        location = f", page {page}" if page is not None else ""
        score_text = f" ({score:.3f})" if isinstance(score, (int, float)) else ""
        lines.append(
            f"{index}. {source}{location}{score_text}\n   Chunk: {chunk_id}\n   {excerpt}"
        )
    return "\n\n".join(lines)


async def _send_query_response(
    update: Update,
    question: str,
    include_citations: bool,
) -> None:
    resp = api_post_text(
        "query",
        {
            "question": question,
            "top_k": 6,
        },
    )
    answer = resp.get("answer", "No answer found.")
    citations = resp.get("citations", []) or []

    # Render answer as Markdown when possible
    try:
        await update.message.reply_text(f"📖 {answer}", parse_mode="Markdown")
    except Exception:
        await update.message.reply_text(f"📖 {answer}")

    if include_citations and citations:
        await update.message.reply_text(
            "🧾 Retrieving source documents...", reply_markup=MAIN_KEYBOARD
        )
        sent_docs = set()

        for idx, citation in enumerate(citations[:5], start=1):
            doc_id = citation.get("document_id")
            source = citation.get("source") or "Unknown"
            page = citation.get("page")
            text = str(citation.get("text") or "").strip()

            if not doc_id or doc_id in sent_docs:
                continue

            sent_docs.add(doc_id)

            try:
                url = f"{RAG_API_URL}/documents/{doc_id}"
                headers = {
                    "Authorization": f"Bearer {RAG_API_KEY}" if RAG_API_KEY else "",
                }
                doc_resp = requests.get(url, headers=headers, timeout=30)
                doc_resp.raise_for_status()
                doc_data = doc_resp.json()

                storage_url = doc_data.get("storage_url")

                # Try provided storage_url first (may be Appwrite download endpoint)
                tried_download = False
                download_headers = {}
                if APPWRITE_PROJECT_ID:
                    download_headers["X-Appwrite-Project"] = APPWRITE_PROJECT_ID
                if APPWRITE_API_KEY:
                    download_headers["X-Appwrite-Key"] = APPWRITE_API_KEY

                if storage_url:
                    location = f" (Page {page})" if page is not None else ""
                    caption = f"📄 {idx}. {source}{location}"
                    try:
                        logger.info(
                            "Attempting download from storage_url: %s", storage_url
                        )
                        file_content = requests.get(
                            storage_url, headers=download_headers or None, timeout=30
                        )
                        file_content.raise_for_status()

                        # Try to extract filename from Content-Disposition header
                        filename = None
                        cd = file_content.headers.get("content-disposition")
                        if cd:
                            m = re.search(r"filename\*?=(?:UTF-8" '|"?)([^";]+)', cd)
                            if m:
                                filename = m.group(1).strip().strip('"')

                        # Fallbacks for filename
                        if not filename:
                            filename = doc_data.get("filename") or doc_data.get(
                                "original_filename"
                            )
                        if not filename:
                            # try last path segment from URL
                            try:
                                filename = (
                                    storage_url.rstrip("/").split("/")[-1].split("?")[0]
                                )
                            except Exception:
                                filename = f"document_{doc_id}"
                        # Ensure extension exists; try to use content-type header
                        content_type = file_content.headers.get("content-type")
                        root, ext = os.path.splitext(filename)
                        if not ext and content_type:
                            ext = MIME_EXTENSION_MAP.get(
                                content_type.split(";")[0].strip()
                            )
                            if ext:
                                filename = f"{filename}{ext}"
                        final_mime = (
                            (content_type.split(";")[0].strip())
                            if content_type
                            else None
                        )

                        # If file is Markdown, render a preview first (truncate to Telegram limits)
                        try:
                            is_md = filename.lower().endswith(".md") or final_mime in (
                                "text/markdown",
                                "text/x-markdown",
                            )
                        except Exception:
                            is_md = False

                        if is_md:
                            try:
                                text_preview = file_content.content.decode("utf-8")
                                preview = (
                                    text_preview
                                    if len(text_preview) <= 3900
                                    else text_preview[:3897] + "..."
                                )
                                await update.message.reply_text(
                                    preview, parse_mode="Markdown"
                                )
                            except Exception:
                                pass

                        # Don't send generic octet-stream files directly — provide a download link instead
                        if final_mime == "application/octet-stream":
                            logger.info(
                                "Skipping send of application/octet-stream for doc %s, sending link instead",
                                doc_id,
                            )
                            await update.message.reply_text(
                                f"🔗 File is a generic binary (application/octet-stream). Download: {storage_url}"
                            )
                        else:
                            file_bytes = io.BytesIO(file_content.content)
                            file_bytes.seek(0)
                            await update.message.reply_document(
                                document=InputFile(file_bytes, filename=filename),
                                caption=caption,
                            )
                        tried_download = True
                    except Exception as storage_exc:
                        logger.warning(
                            f"Could not download from storage {storage_url}: {storage_exc}"
                        )

                # If storage_url failed or not provided, try constructing Appwrite download URL from storage_path
                if not tried_download:
                    storage_path = doc_data.get("storage_path") or doc_data.get(
                        "storage_path"
                    )
                    if storage_path and APPWRITE_ENDPOINT:
                        try:
                            # storage_path format: "{bucket_id}:{file_id}" or similar
                            if ":" in storage_path:
                                bucket_id, file_id = storage_path.split(":", 1)
                            else:
                                # fallback: use configured bucket and storage_path as file_id
                                bucket_id = os.getenv("APPWRITE_BUCKET_ID", "")
                                file_id = storage_path

                            if bucket_id and file_id:
                                metadata_url = f"{APPWRITE_ENDPOINT.rstrip('/')}/storage/buckets/{bucket_id}/files/{file_id}"
                                download_url = f"{APPWRITE_ENDPOINT.rstrip('/')}/storage/buckets/{bucket_id}/files/{file_id}/download"
                                location = f" (Page {page})" if page is not None else ""
                                caption = f"📄 {idx}. {source}{location}"

                                # Fetch metadata first to get filename and mimeType
                                logger.info(
                                    "Fetching Appwrite metadata: %s", metadata_url
                                )
                                meta_resp = requests.get(
                                    metadata_url,
                                    headers=download_headers or None,
                                    timeout=20,
                                )
                                if meta_resp.status_code == 200:
                                    meta = meta_resp.json()
                                    logger.debug(
                                        "Appwrite metadata for %s: %s", file_id, meta
                                    )
                                    meta_name = meta.get("name") or meta.get("$id")
                                    meta_mime = meta.get("mimeType") or meta.get(
                                        "mime_type"
                                    )
                                else:
                                    logger.warning(
                                        "Appwrite metadata request failed %s: %s",
                                        metadata_url,
                                        meta_resp.status_code,
                                    )
                                    meta = {}
                                    meta_name = None
                                    meta_mime = None

                                logger.info(
                                    "Attempting Appwrite download URL: %s", download_url
                                )
                                file_content = requests.get(
                                    download_url,
                                    headers=download_headers or None,
                                    timeout=30,
                                )
                                file_content.raise_for_status()

                                # Determine filename: header > metadata > file_id
                                filename = None
                                cd = file_content.headers.get("content-disposition")
                                if cd:
                                    m = re.search(
                                        r"filename\*?=(?:UTF-8''|\"?)([^\";]+)", cd
                                    )
                                    if m:
                                        filename = m.group(1).strip().strip('"')
                                if not filename:
                                    filename = (
                                        meta_name
                                        or doc_data.get("filename")
                                        or doc_data.get("original_filename")
                                    )
                                if not filename:
                                    filename = f"{file_id}"

                                # Ensure extension exists
                                content_type = (
                                    file_content.headers.get("content-type")
                                    or meta_mime
                                )
                                root, ext = os.path.splitext(filename)
                                if not ext and content_type:
                                    ext = MIME_EXTENSION_MAP.get(
                                        content_type.split(";")[0].strip()
                                    )
                                    if ext:
                                        filename = f"{filename}{ext}"

                                final_mime = (
                                    (content_type.split(";")[0].strip())
                                    if content_type
                                    else None
                                )

                                # If Markdown, render preview first
                                try:
                                    is_md = filename.lower().endswith(
                                        ".md"
                                    ) or final_mime in (
                                        "text/markdown",
                                        "text/x-markdown",
                                    )
                                except Exception:
                                    is_md = False
                                if is_md:
                                    try:
                                        text_preview = file_content.content.decode(
                                            "utf-8"
                                        )
                                        preview = (
                                            text_preview
                                            if len(text_preview) <= 3900
                                            else text_preview[:3897] + "..."
                                        )
                                        await update.message.reply_text(
                                            preview, parse_mode="Markdown"
                                        )
                                    except Exception:
                                        pass

                                if final_mime == "application/octet-stream":
                                    logger.info(
                                        "Skipping send of application/octet-stream for doc %s, sending link instead",
                                        file_id,
                                    )
                                    await update.message.reply_text(
                                        f"🔗 File is a generic binary (application/octet-stream). Download: {download_url}"
                                    )
                                else:
                                    file_bytes = io.BytesIO(file_content.content)
                                    file_bytes.seek(0)
                                    await update.message.reply_document(
                                        document=InputFile(
                                            file_bytes, filename=filename
                                        ),
                                        caption=caption,
                                    )
                                tried_download = True
                        except Exception as appwrite_exc:
                            logger.warning(
                                f"Could not download from Appwrite {storage_path}: {appwrite_exc}"
                            )

                # If still not downloaded, fall back to text excerpt as Markdown
                if not tried_download and text:
                    location = f" (Page {page})" if page is not None else ""
                    caption = f"📄 {idx}. {source}{location} (text excerpt)"
                    md_content = f"# Citation: {source}{location}\n\n{ text }\n"
                    filename = f"excerpt_{doc_id}.md"
                    file_bytes = io.BytesIO(md_content.encode("utf-8"))
                    file_bytes.seek(0)
                    await update.message.reply_document(
                        document=InputFile(file_bytes, filename=filename),
                        caption=caption,
                    )
            except Exception as exc:
                logger.warning(f"Could not retrieve document {doc_id}: {exc}")
                continue


async def cmd_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Save a text entry: /text title|content."""
    if not await require_authorized(update):
        return

    args = " ".join(context.args) if context.args else ""
    if "|" not in args:
        await update.message.reply_text(
            "Usage: `/text title|content`", parse_mode="Markdown"
        )
        return

    parts = args.split("|", 1)
    title = parts[0].strip()
    content = parts[1].strip()

    try:
        await update.message.reply_text("📝 Saving text entry...")
        resp = api_post_text(
            "documents/text",
            {
                "title": title,
                "content": content,
                "source_type": "telegram",
            },
        )
        await update.message.reply_text(
            f"✅ Saved!\nDocument ID: `{resp.get('document_id', 'N/A')}`",
            parse_mode="Markdown",
        )
    except Exception as exc:
        await update.message.reply_text(f"❌ Error: {exc}")


async def cmd_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Query the knowledge base: /query question."""
    if not await require_authorized(update):
        return

    question = " ".join(context.args) if context.args else ""
    if not question:
        await update.message.reply_text(
            "Usage: `/query your question`", parse_mode="Markdown"
        )
        return

    try:
        await update.message.reply_text("🔍 Searching...")
        resp = api_post_text(
            "query",
            {
                "question": question,
                "top_k": 6,
            },
        )
        answer = resp.get("answer", "No answer found.")
        try:
            await update.message.reply_text(f"📖 {answer}", parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(f"📖 {answer}")
    except Exception as exc:
        await update.message.reply_text(f"❌ Error: {exc}")


async def handle_voice_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle voice messages: transcribe, query, reply with audio."""
    if not await require_authorized(update):
        return

    try:
        await update.message.reply_text("🎤 Processing voice message...")

        # Download voice file
        voice = update.message.voice
        file = await context.bot.get_file(voice.file_id)
        bio = io.BytesIO()
        await file.download_to_memory(bio)
        bio.seek(0)

        # Query audio endpoint
        resp = api_post_file(
            "query/audio",
            bio.read(),
            "voice.oga",
            "audio/ogg",
            extra_data={"top_k": "6"},
        )

        # Extract response
        transcript = resp.get("transcript", "")
        answer = resp.get("answer", "No answer found.")
        audio_base64 = resp.get("audio_base64")
        audio_storage_url = resp.get("audio_storage_url")
        audio_error = resp.get("audio_error")

        # Send text answer
        msg = f"🎤 **You said:** {transcript}\n\n📖 **Answer:** {answer}"
        await update.message.reply_text(msg, parse_mode="Markdown")

        # Send audio reply if available
        if audio_base64:
            audio_bytes = __import__("base64").b64decode(audio_base64)
            await update.message.reply_voice(
                voice=io.BytesIO(audio_bytes),
                caption="🔊 Audio reply",
            )
        elif audio_storage_url:
            await update.message.reply_text(f"🔊 Audio reply: {audio_storage_url}")
        elif audio_error:
            await update.message.reply_text(f"⚠️ Audio reply failed: {audio_error}")

    except Exception as exc:
        logger.exception("Voice message failed")
        await update.message.reply_text(f"❌ Error: {exc}")


async def handle_audio_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle audio file uploads."""
    if not await require_authorized(update):
        return

    try:
        await update.message.reply_text("📎 Processing audio file...")

        # Download audio file
        audio = update.message.audio
        file = await context.bot.get_file(audio.file_id)
        bio = io.BytesIO()
        await file.download_to_memory(bio)
        bio.seek(0)

        # Upload to knowledge base
        resp = api_post_file(
            "documents/audio",
            bio.read(),
            audio.file_name or "audio.mp3",
            audio.mime_type or "audio/mpeg",
        )

        doc_id = resp.get("document_id", "N/A")
        msg = resp.get("message", "Uploaded successfully.")
        await update.message.reply_text(
            f"✅ {msg}\nID: `{doc_id}`", parse_mode="Markdown"
        )

    except Exception as exc:
        logger.exception("Audio upload failed")
        await update.message.reply_text(f"❌ Error: {exc}")


async def handle_document_upload(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle document file uploads (PDF, image, etc.)."""
    if not await require_authorized(update):
        return

    try:
        await update.message.reply_text("📎 Processing document...")

        # Download file
        doc = update.message.document
        file = await context.bot.get_file(doc.file_id)
        bio = io.BytesIO()
        await file.download_to_memory(bio)
        bio.seek(0)

        # Upload to knowledge base
        resp = api_post_file(
            "documents/upload",
            bio.read(),
            doc.file_name or "document",
            doc.mime_type or "application/octet-stream",
        )

        doc_id = resp.get("document_id", "N/A")
        msg = resp.get("message", "Uploaded successfully.")
        await update.message.reply_text(
            f"✅ {msg}\nID: `{doc_id}`", parse_mode="Markdown"
        )

    except Exception as exc:
        logger.exception("Document upload failed")
        await update.message.reply_text(f"❌ Error: {exc}")


async def handle_text_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle plain text queries."""
    if not await require_authorized(update):
        return

    question = update.message.text
    if not question:
        return

    pending_mode = context.user_data.pop(QUERY_MODE_KEY, None)

    if question == "Start":
        await start(update, context)
        return
    if question == "Help":
        await help_command(update, context)
        return
    if question == "Query":
        await query_command(update, context)
        return
    if question == "Query Doc":
        await query_doc_command(update, context)
        return
    if question == "Text":
        await update.message.reply_text(
            "Use /text title|content to save a note.", reply_markup=MAIN_KEYBOARD
        )
        return
    if question == "Stats":
        await stats_command(update, context)
        return
    if question == "Reset":
        await reset_command(update, context)
        return
    if question == "ID":
        await id_command(update, context)
        return

    if pending_mode == "query_doc":
        try:
            await update.message.reply_text("🔎 Searching with citations...")
            await _send_query_response(update, question, include_citations=True)
        except Exception as exc:
            logger.exception("Cited query failed")
            await update.message.reply_text(f"❌ Error: {exc}")
        return

    if pending_mode == "query" or pending_mode is None:
        try:
            await update.message.reply_text("🔍 Searching...")
            await _send_query_response(update, question, include_citations=False)
        except Exception as exc:
            logger.exception("Query failed")
            await update.message.reply_text(f"❌ Error: {exc}")
        return


def main() -> None:
    """Start the bot."""
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN environment variable is not set.")

    logger.info(f"Starting Telegram bot... API: {RAG_API_URL}")
    if ALLOWED_USERS:
        logger.info("Telegram allowlist enabled for %s users", len(ALLOWED_USERS))
    else:
        logger.warning("Telegram allowlist is empty; all users will be rejected")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("ask", ask_command))
    app.add_handler(CommandHandler("query_doc", query_doc_command))
    app.add_handler(CommandHandler("text", cmd_text))
    app.add_handler(CommandHandler("query", cmd_query))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("id", id_command))
    app.add_handler(
        CallbackQueryHandler(server_status_callback, pattern="^server_status$")
    )

    # Message handlers
    app.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    app.add_handler(MessageHandler(filters.AUDIO, handle_audio_file))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document_upload))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message)
    )

    logger.info("Bot started. Polling for messages...")
    app.run_polling()


if __name__ == "__main__":
    main()
