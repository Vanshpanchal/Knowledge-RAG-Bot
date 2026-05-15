# Knowledge RAG Bot

Production-ready Telegram bot + modular RAG API for personal knowledge retrieval.
This project lets you upload documents/audio, process them through a RAG pipeline,
and ask grounded questions from Telegram or API clients.

## What This Project Includes

- Telegram bot interface (`bot.py`)
- FastAPI backend (`app/`)
- Document/audio ingestion and processing pipeline
- Retrieval + grounded answer generation
- Optional GitHub workflow templates in `.github/workflow-templates/`

## Core Features

- Telegram commands and keyboard workflow for querying and uploads
- File upload support (PDF, images, text, audio)
- Voice query flow: speech input -> RAG answer -> optional audio reply
- Citation-aware responses (`query_doc` mode)
- Config-driven provider selection for storage, OCR, embeddings, and LLM

## High-Level RAG Flow

1. Upload document/audio
2. Validate size, type, and extension
3. Persist file to storage
4. Save document metadata to MongoDB
5. Run async ingestion pipeline
6. Extract text (OCR fallback when needed)
7. Clean and normalize text
8. Chunk content semantically
9. Generate embeddings
10. Save chunks + vectors
11. Query with layered retrieval
12. Return grounded answer with optional citations

## API Endpoints

- `GET /health`
- `GET /ready`
- `POST /api/v1/documents/upload`
- `POST /api/v1/documents/audio`
- `POST /api/v1/documents/text`
- `GET /api/v1/documents/{document_id}`
- `POST /api/v1/query`
- `POST /api/v1/query/audio`

## Telegram Bot Commands

- `/start` and `/help`: show usage
- `/query <question>`: standard answer flow
- `/query_doc <question>`: answer with citations and downloadable summary
- `/text <note>`: save text into the knowledge base
- `/stats`: bot status placeholder
- `/reset`: reset local conversation state
- `/id`: show Telegram numeric user ID

Supported Telegram interactions:

- Send a document file to ingest knowledge
- Send an audio file to ingest/query
- Send a voice message to query via speech

## Prerequisites

- Python 3.10+
- MongoDB (local or Atlas)
- Telegram Bot token
- API keys based on chosen providers

## Local Setup

```bash
git clone <your-repo-url> knowledge-rag-bot
cd knowledge-rag-bot
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r app/requirements.txt
```

Create environment file:

```bash
cp .env.example .env
```

On Windows cmd:

```bat
copy .env.example .env
```

Fill `.env` with your real values before running.

## Minimum Environment Variables

For Telegram bot:

- `TELEGRAM_TOKEN`
- `RAG_API_KEY`
- `CORTEX_BASE_URL` (for hosted API) or local API URL
- `ALLOWED_USERS` (comma-separated Telegram user IDs)

For API runtime:

- `MONGODB_URI`
- `DATABASE_NAME`
- `API_KEYS` (comma-separated API keys accepted by backend)
- `GEMINI_API_KEY` and/or other provider keys based on configuration

## Run the API

```bash
uvicorn app.main:app --reload
```

Default URL: `http://127.0.0.1:8000`

## Run the Telegram Bot

```bash
python bot.py
```

The bot sends requests to `${CORTEX_BASE_URL}/api/v1`.

## Security Notes

- Never commit `.env`, tokens, or credentials
- Keep production keys in secret managers or platform secrets
- Restrict bot access with `ALLOWED_USERS`
- Rotate API keys regularly

## CI/CD Templates (Optional)

This repository ships workflow templates only.
No GitHub Actions run until templates are copied into `.github/workflows/`.

See `.github/README.md` and `.github/workflow-templates/README.md` for activation steps.




