# Multi-stage Dockerfile for Telegram bot
FROM python:3.14-slim AS builder

WORKDIR /build

# Create a virtual environment and install dependencies into it
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Copy requirements
COPY requirements.txt .

# Install dependencies into the virtual environment
RUN pip install --no-cache-dir -r requirements.txt


# Final stage
FROM python:3.14-slim

WORKDIR /app

# Copy Python virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set PATH to use the virtual environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Copy bot and config
COPY bot.py .
COPY app ./app

# Create non-root user for security
RUN useradd -m -u 1000 botuser && chown -R botuser:botuser /app
USER botuser

# Copy startup script
COPY --chown=botuser:botuser start.sh .
RUN chmod +x /app/start.sh

# Health check (optional, for orchestration)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Run API + bot in one container for Render
CMD ["/app/start.sh"]
