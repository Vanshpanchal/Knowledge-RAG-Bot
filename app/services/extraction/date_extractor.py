"""Date extraction from document content."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from dateutil import parser as date_parser


class DateExtractor:
    """Extract dates and temporal references from document text."""

    # Common date patterns
    DATE_PATTERNS = [
        # ISO format: 2024-01-15
        (r"(\d{4})-(\d{1,2})-(\d{1,2})", "%Y-%m-%d"),
        # US format: 01/15/2024
        (r"(\d{1,2})/(\d{1,2})/(\d{4})", "%m/%d/%Y"),
        # European format: 15.01.2024
        (r"(\d{1,2})\.(\d{1,2})\.(\d{4})", "%d.%m.%Y"),
        # Long format: January 15, 2024 or 15 January 2024
        (
            r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})",
            None,
        ),
        (
            r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})",
            None,
        ),
        # Month/Year: January 2024
        (
            r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})",
            None,
        ),
    ]

    # Temporal keywords that indicate when content was created/modified
    TEMPORAL_KEYWORDS = [
        r"\b(created|generated|produced|written|authored|posted|published|on|at|date)\b",
        r"\b(yesterday|today|tomorrow|last\s+(?:week|month|year))\b",
        r"\b(morning|afternoon|evening|night)\b",
    ]

    @staticmethod
    def extract_dates(text: str) -> List[datetime]:
        """Extract all dates found in text.

        Args:
            text: Document text to search for dates.

        Returns:
            List of datetime objects found in the text.
        """
        dates = []

        for pattern, format_str in DateExtractor.DATE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    if format_str:
                        date_str = match.group(0)
                        parsed_date = datetime.strptime(date_str, format_str)
                    else:
                        date_str = match.group(0)
                        parsed_date = date_parser.parse(date_str)

                    # Localize to UTC if naive
                    if parsed_date.tzinfo is None:
                        parsed_date = parsed_date.replace(tzinfo=timezone.utc)

                    dates.append(parsed_date)
                except (ValueError, TypeError):
                    continue

        return sorted(list(set(dates)))  # Remove duplicates and sort

    @staticmethod
    def find_document_date(text: str) -> Optional[datetime]:
        """Extract the most likely creation/modification date from document.

        Tries to find the first date mentioned with temporal context keywords.

        Args:
            text: Document text.

        Returns:
            Most likely document date, or None if not found.
        """
        # Split text into sentences
        sentences = re.split(r"[.!?]\s+", text)

        for sentence in sentences:
            # Check if sentence has temporal keywords
            has_temporal = any(
                re.search(pattern, sentence, re.IGNORECASE)
                for pattern in DateExtractor.TEMPORAL_KEYWORDS
            )

            if has_temporal:
                # Try to find dates in this sentence
                dates = DateExtractor.extract_dates(sentence)
                if dates:
                    return dates[0]

        # Fall back to first date found in document
        all_dates = DateExtractor.extract_dates(text)
        if all_dates:
            return all_dates[0]

        return None

    @staticmethod
    def infer_metadata_dates(
        text: str,
        uploaded_timestamp: datetime,
        filename: str = "",
    ) -> Dict[str, Any]:
        """Infer date metadata from document.

        Args:
            text: Document text.
            uploaded_timestamp: When the document was uploaded.
            filename: Document filename (may contain date patterns).

        Returns:
            Dictionary with date metadata.
        """
        metadata = {
            "uploaded_at": uploaded_timestamp,
            "extracted_dates": DateExtractor.extract_dates(text),
            "document_date": DateExtractor.find_document_date(text),
        }

        # Try to extract date from filename
        if filename:
            filename_dates = DateExtractor.extract_dates(filename)
            if filename_dates:
                metadata["filename_date"] = filename_dates[0]

        # Use the most reliable date source
        # Priority: document_date > filename_date > uploaded_at
        if metadata["document_date"]:
            metadata["date"] = metadata["document_date"]
        elif metadata.get("filename_date"):
            metadata["date"] = metadata["filename_date"]
        else:
            metadata["date"] = uploaded_timestamp

        return metadata
