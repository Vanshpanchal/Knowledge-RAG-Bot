"""Temporal query processing for handling date and time-related questions."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta


class TemporalExpression:
    """Represents a temporal constraint parsed from a query."""

    def __init__(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        relative_term: Optional[str] = None,
        is_relative: bool = False,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.relative_term = relative_term
        self.is_relative = is_relative

    def to_filter(self) -> Dict[str, Any]:
        """Convert temporal expression to MongoDB filter."""
        if not self.start_date and not self.end_date:
            return {}

        filter_dict = {}

        if self.start_date and self.end_date:
            filter_dict["metadata.date"] = {
                "$gte": self.start_date,
                "$lte": self.end_date,
            }
        elif self.start_date:
            filter_dict["metadata.date"] = {"$gte": self.start_date}
        elif self.end_date:
            filter_dict["metadata.date"] = {"$lte": self.end_date}

        return filter_dict


class TemporalProcessor:
    """Process and extract temporal information from queries."""

    # Regex patterns for temporal expressions
    TEMPORAL_PATTERNS = {
        "yesterday": r"\byesterday\'?s?\b",  # yesterday, yesterdays, yesterday's
        "today": r"\btoday\'?s?\b",  # today, todays, today's
        "tomorrow": r"\btomorrow\'?s?\b",  # tomorrow, tomorrows, tomorrow's
        "last_week": r"\blast\s+week\b",
        "last_month": r"\blast\s+month\b",
        "last_year": r"\blast\s+year\b",
        "last_n_days": r"\blast\s+(\d+)\s+days?\b",
        "last_n_weeks": r"\blast\s+(\d+)\s+weeks?\b",
        "last_n_months": r"\blast\s+(\d+)\s+months?\b",
        "past_n_days": r"\bpast\s+(\d+)\s+days?\b",
        "past_n_weeks": r"\bpast\s+(\d+)\s+weeks?\b",
        "past_n_months": r"\bpast\s+(\d+)\s+months?\b",
        "this_week": r"\bthis\s+week\b",
        "this_month": r"\bthis\s+month\b",
        "this_year": r"\bthis\s+year\b",
        "all_time": r"\ball\s+time\b",
    }

    @staticmethod
    def get_current_time() -> datetime:
        """Get current time in UTC."""
        return datetime.now(timezone.utc)

    @classmethod
    def parse_temporal_expression(cls, query: str) -> Optional[TemporalExpression]:
        """Parse temporal expressions from query.

        Args:
            query: User question that may contain temporal references.

        Returns:
            TemporalExpression if temporal term found, None otherwise.
        """
        query_lower = query.lower()
        now = cls.get_current_time()

        # Check for "all time" - no temporal constraints
        if re.search(cls.TEMPORAL_PATTERNS["all_time"], query_lower):
            return TemporalExpression()

        # Check for "yesterday"
        if re.search(cls.TEMPORAL_PATTERNS["yesterday"], query_lower):
            start = now - timedelta(days=1)
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1) - timedelta(seconds=1)
            return TemporalExpression(
                start_date=start, end_date=end, relative_term="yesterday"
            )

        # Check for "today"
        if re.search(cls.TEMPORAL_PATTERNS["today"], query_lower):
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = now
            return TemporalExpression(
                start_date=start, end_date=end, relative_term="today"
            )

        # Check for "tomorrow"
        if re.search(cls.TEMPORAL_PATTERNS["tomorrow"], query_lower):
            start = now + timedelta(days=1)
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1) - timedelta(seconds=1)
            return TemporalExpression(
                start_date=start, end_date=end, relative_term="tomorrow"
            )

        # Check for "this week"
        if re.search(cls.TEMPORAL_PATTERNS["this_week"], query_lower):
            # Start of current week (Monday)
            days_since_monday = now.weekday()
            start = now - timedelta(days=days_since_monday)
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = now
            return TemporalExpression(
                start_date=start, end_date=end, relative_term="this_week"
            )

        # Check for "last week"
        if re.search(cls.TEMPORAL_PATTERNS["last_week"], query_lower):
            days_since_monday = now.weekday()
            # Start of this week
            this_week_start = now - timedelta(days=days_since_monday)
            # Start of last week
            start = this_week_start - timedelta(weeks=1)
            end = this_week_start - timedelta(seconds=1)
            return TemporalExpression(
                start_date=start, end_date=end, relative_term="last_week"
            )

        # Check for "this month"
        if re.search(cls.TEMPORAL_PATTERNS["this_month"], query_lower):
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end = now
            return TemporalExpression(
                start_date=start, end_date=end, relative_term="this_month"
            )

        # Check for "last month"
        if re.search(cls.TEMPORAL_PATTERNS["last_month"], query_lower):
            start = (now.replace(day=1) - timedelta(days=1)).replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )
            end = now.replace(day=1) - timedelta(seconds=1)
            return TemporalExpression(
                start_date=start, end_date=end, relative_term="last_month"
            )

        # Check for "this year"
        if re.search(cls.TEMPORAL_PATTERNS["this_year"], query_lower):
            start = now.replace(
                month=1, day=1, hour=0, minute=0, second=0, microsecond=0
            )
            end = now
            return TemporalExpression(
                start_date=start, end_date=end, relative_term="this_year"
            )

        # Check for "last year"
        if re.search(cls.TEMPORAL_PATTERNS["last_year"], query_lower):
            last_year = now.year - 1
            start = now.replace(
                year=last_year,
                month=1,
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
            end = now.replace(
                year=last_year,
                month=12,
                day=31,
                hour=23,
                minute=59,
                second=59,
                microsecond=0,
            )
            return TemporalExpression(
                start_date=start, end_date=end, relative_term="last_year"
            )

        # Check for "last N days"
        match = re.search(cls.TEMPORAL_PATTERNS["last_n_days"], query_lower)
        if match:
            n_days = int(match.group(1))
            start = now - timedelta(days=n_days)
            end = now
            return TemporalExpression(
                start_date=start, end_date=end, relative_term=f"last_{n_days}_days"
            )

        # Check for "last N weeks"
        match = re.search(cls.TEMPORAL_PATTERNS["last_n_weeks"], query_lower)
        if match:
            n_weeks = int(match.group(1))
            start = now - timedelta(weeks=n_weeks)
            end = now
            return TemporalExpression(
                start_date=start, end_date=end, relative_term=f"last_{n_weeks}_weeks"
            )

        # Check for "last N months"
        match = re.search(cls.TEMPORAL_PATTERNS["last_n_months"], query_lower)
        if match:
            n_months = int(match.group(1))
            start = now - relativedelta(months=n_months)
            end = now
            return TemporalExpression(
                start_date=start, end_date=end, relative_term=f"last_{n_months}_months"
            )

        # Check for "past N days"
        match = re.search(cls.TEMPORAL_PATTERNS["past_n_days"], query_lower)
        if match:
            n_days = int(match.group(1))
            start = now - timedelta(days=n_days)
            end = now
            return TemporalExpression(
                start_date=start, end_date=end, relative_term=f"past_{n_days}_days"
            )

        # Check for "past N weeks"
        match = re.search(cls.TEMPORAL_PATTERNS["past_n_weeks"], query_lower)
        if match:
            n_weeks = int(match.group(1))
            start = now - timedelta(weeks=n_weeks)
            end = now
            return TemporalExpression(
                start_date=start, end_date=end, relative_term=f"past_{n_weeks}_weeks"
            )

        # Check for "past N months"
        match = re.search(cls.TEMPORAL_PATTERNS["past_n_months"], query_lower)
        if match:
            n_months = int(match.group(1))
            start = now - relativedelta(months=n_months)
            end = now
            return TemporalExpression(
                start_date=start, end_date=end, relative_term=f"past_{n_months}_months"
            )

        # Try to parse absolute dates (e.g., "2024-01-15", "January 15, 2024")
        date_patterns = [
            r"(\d{4}-\d{2}-\d{2})",  # YYYY-MM-DD
            r"(\d{1,2}/\d{1,2}/\d{2,4})",  # MM/DD/YYYY or DD/MM/YYYY
            r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}",  # Month Day, Year
        ]

        for pattern in date_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    date_str = match.group(1)
                    parsed_date = date_parser.parse(date_str)
                    start = parsed_date.replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                    end = parsed_date.replace(
                        hour=23, minute=59, second=59, microsecond=0
                    )
                    return TemporalExpression(
                        start_date=start, end_date=end, relative_term=date_str
                    )
                except (ValueError, TypeError):
                    continue

        return None

    @classmethod
    def remove_temporal_terms(cls, query: str) -> str:
        """Remove temporal terms from query for cleaner keyword search.

        Args:
            query: Original query.

        Returns:
            Query with temporal terms removed.
        """
        query_lower = query.lower()

        # Remove temporal patterns
        for pattern in cls.TEMPORAL_PATTERNS.values():
            query = re.sub(pattern, "", query, flags=re.IGNORECASE)

        # Clean up extra whitespace
        query = re.sub(r"\s+", " ", query).strip()

        return query
