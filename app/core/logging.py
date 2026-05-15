"""Centralized logging configuration."""
import logging
import sys


def setup_logging(show_logs: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.INFO if show_logs else logging.WARNING
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Suppress httpx logs unless logs are enabled
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


def configure_logging(level: str = "INFO") -> None:
    """Configure logging with a specific level string."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    setup_logging(show_logs=(numeric_level == logging.INFO))