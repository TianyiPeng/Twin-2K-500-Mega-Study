"""Utility functions for processing Qualtrics QSF question types."""

import html
import logging
import re
from typing import Any, Dict

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def strip_html_table(text: str) -> str:
    """Strip HTML table into readable key-value format or remove all tags."""
    try:
        processed_text = html.unescape(text)
        soup = BeautifulSoup(processed_text, "html.parser")

        # Try parsing table headers and values
        headers = [th.get_text(strip=True) for th in soup.find_all("th")]
        first_row = soup.find("tbody")
        if not headers or not first_row:
            raise ValueError("No table detected")

        row = first_row.find("tr")
        if not row:
            raise ValueError("No <tr> in table")

        values = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(headers) != len(values):
            raise ValueError("Header/value mismatch")

        # Format as text output
        parsed_dict = dict(zip(headers, values, strict=False))
        return 0, ". ".join(f"{k}: {v}" for k, v in parsed_dict.items())

    except Exception:
        return 1, ""


def strip_html(text: str) -> str:
    """Strip HTML tags from text and normalize whitespace."""
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]*>", " ", text)
    # Replace &nbsp; with space
    text = text.replace("&nbsp;", " ")
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def default_order(options_id: list) -> bool:
    """Check if the options are in default order."""
    return options_id == list(str(i) for i in range(1, len(options_id) + 1))


def get_common_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Get common settings for all question types."""
    return {
        "Selector": payload.get("Selector", ""),
        "SubSelector": payload.get("SubSelector", ""),
        "ForceResponse": payload.get("Validation", {}).get("Settings", {}).get("ForceResponse", ""),
    }
