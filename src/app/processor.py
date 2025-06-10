"""Processor module for alternative web sentiment analysis (e.g., headlines, web content)."""

from typing import Any

from textblob import TextBlob

from app.utils.setup_logger import setup_logger
from app.utils.types import validate_dict

# Initialize logger
logger = setup_logger(__name__)


def process(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Process a batch of messages for sentiment analysis.

    Each message must contain a 'text' field.
    Sentiment polarity and label are added to the message.

    Parameters
    ----------
    payloads : list[dict[str, Any]]
        List of incoming messages to process.

    Returns
    -------
    list[dict[str, Any]]
        Enriched messages with sentiment analysis results.

    """
    results: list[dict[str, Any]] = []

    for item in payloads:
        if not validate_dict(item, ["text"]):
            logger.warning("⚠️ Skipping message: missing 'text' field: %s", item)
            continue

        text = item.get("text", "")
        try:
            blob = TextBlob(str(text))
            sentiment = blob.sentiment  # type: ignore[union-attr]
            polarity = float(getattr(sentiment, "polarity", 0.0))

            if polarity > 0.1:
                label = "positive"
            elif polarity < -0.1:
                label = "negative"
            else:
                label = "neutral"

            item["sentiment_score"] = polarity
            item["sentiment_label"] = label
            results.append(item)
        except Exception as e:
            logger.exception("❌ Sentiment analysis failed for input: %s | Error: %s", text, e)

    return results
