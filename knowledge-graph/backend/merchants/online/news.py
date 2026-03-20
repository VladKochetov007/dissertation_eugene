"""
News Aggregation Merchant Agent — RSS feed ingestion with entity extraction.

Collects news articles from configurable RSS feeds, parses their content,
and extracts keywords and named entities for linking to the knowledge graph.
News is the narrative layer of reality — what people SAY is happening —
and forms essential observational data for causal analysis.

Data collected:
- Article metadata (title, author, publication date, source)
- Article content / summary
- Extracted keywords and named entities
- Feed-level metadata

Pearl Level: 1 (Association) — observational data from news sources.

Usage:
    from graph.store import KnowledgeGraphStore
    from merchants.online.news import NewsMerchant

    store = KnowledgeGraphStore()
    merchant = NewsMerchant(
        store=store,
        feeds=[
            "https://feeds.reuters.com/reuters/topNews",
            "https://feeds.bbci.co.uk/news/rss.xml",
        ],
    )
    await merchant.start()
"""

from __future__ import annotations

import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Optional

from graph.entities import DataSourceType, VariableType
from graph.store import KnowledgeGraphStore

from ..base import (
    CollectionResult,
    MerchantAgent,
    MerchantConfig,
    PearlLevel,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_POLL_INTERVAL = 600.0  # 10 minutes
DEFAULT_RATE_LIMIT = 10.0  # requests per minute

# Default news feeds — a balanced selection of major sources
DEFAULT_FEEDS = [
    "https://feeds.reuters.com/reuters/topNews",
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://feeds.npr.org/1001/rss.xml",
]

# Simple keyword extraction stopwords
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "it", "its", "this",
    "that", "these", "those", "he", "she", "they", "we", "you", "i", "my",
    "his", "her", "their", "our", "your", "not", "no", "if", "as", "so",
    "up", "out", "about", "into", "over", "after", "than", "also", "just",
    "more", "some", "any", "all", "very", "s", "t", "re", "ve", "ll", "d",
    "said", "says", "new", "us", "who", "what", "when", "where", "how",
})


# ---------------------------------------------------------------------------
# News Merchant
# ---------------------------------------------------------------------------


class NewsMerchant(MerchantAgent):
    """Merchant agent for news aggregation from RSS feeds.

    Collects articles from configured RSS feeds, extracts keywords and
    named entities, and creates Variable entities representing news topics
    and their associated sources.

    Attributes:
        feeds: List of RSS feed URLs to monitor.
        max_articles_per_feed: Maximum articles to process per feed per cycle.
        _executor: Thread pool for synchronous RSS parsing.
        _seen_articles: Set of article hashes to avoid duplicate processing.
    """

    def __init__(
        self,
        store: KnowledgeGraphStore,
        feeds: Optional[list[str]] = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        rate_limit: float = DEFAULT_RATE_LIMIT,
        max_articles_per_feed: int = 20,
    ) -> None:
        """Initialize the news aggregation merchant.

        Args:
            store: Knowledge graph store for data ingestion.
            feeds: List of RSS feed URLs. If None, uses default feeds.
            poll_interval: Seconds between polling cycles.
            rate_limit: Maximum requests per minute.
            max_articles_per_feed: Max articles to process per feed.
        """
        config = MerchantConfig(
            name="news-aggregator",
            source_type=DataSourceType.NEWS,
            pearl_level=PearlLevel.ASSOCIATION,
            poll_interval_seconds=poll_interval,
            rate_limit_requests_per_minute=rate_limit,
            metadata={"max_articles_per_feed": max_articles_per_feed},
        )
        super().__init__(config=config, store=store)

        self.feeds = feeds if feeds is not None else list(DEFAULT_FEEDS)
        self.max_articles_per_feed = max_articles_per_feed
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._seen_articles: set[str] = set()

        # Register data source
        self._data_source = self._register_data_source(
            name="News Aggregation (RSS)",
            url=None,
            schema_info={
                "type": "news_feed",
                "format": "rss",
                "data_fields": [
                    "title",
                    "link",
                    "published",
                    "summary",
                    "author",
                    "keywords",
                    "source_feed",
                ],
            },
        )

    # ------------------------------------------------------------------
    # Pipeline implementation
    # ------------------------------------------------------------------

    async def discover(self) -> list[str]:
        """Return the configured list of RSS feed URLs.

        Discovery for news is the list of feeds. Dynamic feed discovery
        (e.g., finding new relevant feeds) could be added later.

        Returns:
            List of RSS feed URLs to collect from.
        """
        logger.info("News: monitoring %d RSS feeds.", len(self.feeds))
        return list(self.feeds)

    async def collect(self, targets: list[str]) -> list[CollectionResult]:
        """Parse RSS feeds and collect article data.

        Uses feedparser in a thread pool to parse each RSS feed,
        extracting article metadata and content.

        Args:
            targets: List of RSS feed URLs.

        Returns:
            List of CollectionResult objects, one per article.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        results: list[CollectionResult] = []

        for feed_url in targets:
            try:
                await self._enforce_rate_limit()
                articles = await loop.run_in_executor(
                    self._executor, self._parse_feed, feed_url
                )

                for article in articles[: self.max_articles_per_feed]:
                    # Deduplicate by content hash
                    article_hash = self._hash_article(article)
                    if article_hash in self._seen_articles:
                        continue
                    self._seen_articles.add(article_hash)

                    # Extract keywords
                    text = f"{article.get('title', '')} {article.get('summary', '')}"
                    keywords = self._extract_keywords(text)

                    result = CollectionResult(
                        target=feed_url,
                        data={
                            "title": article.get("title", ""),
                            "link": article.get("link", ""),
                            "published": article.get("published", ""),
                            "summary": article.get("summary", ""),
                            "author": article.get("author", ""),
                            "source_feed": feed_url,
                            "keywords": keywords,
                            "article_hash": article_hash,
                        },
                        provenance=self._build_provenance(
                            source_url=article.get("link", feed_url),
                            collection_method="rss_feed",
                            confidence=0.7,
                            raw_record_count=1,
                        ),
                    )
                    results.append(result)

            except Exception as exc:
                logger.warning("News: failed to parse feed %s: %s", feed_url, exc)
                results.append(
                    CollectionResult(
                        target=feed_url,
                        data={},
                        provenance=self._build_provenance(
                            source_url=feed_url,
                            confidence=0.0,
                        ),
                        is_valid=False,
                        validation_errors=[f"Feed parse error: {exc}"],
                    )
                )

        logger.info("News: collected %d articles from %d feeds.", len(results), len(targets))
        return results

    async def validate(self, results: list[CollectionResult]) -> list[CollectionResult]:
        """Validate collected news articles.

        Checks for required fields (title, link) and reasonable content.

        Args:
            results: Collection results to validate.

        Returns:
            Results with is_valid and validation_errors updated.
        """
        for result in results:
            if result.validation_errors:
                continue

            errors: list[str] = []
            data = result.data

            if not data.get("title"):
                errors.append("Missing article title.")
            if not data.get("link"):
                errors.append("Missing article link.")

            # Title should be reasonable length
            title = data.get("title", "")
            if len(title) > 1000:
                errors.append(f"Title too long ({len(title)} chars).")

            result.validation_errors = errors
            result.is_valid = len(errors) == 0

        valid_count = sum(1 for r in results if r.is_valid)
        logger.info("News: validated %d/%d articles.", valid_count, len(results))
        return results

    async def ingest(self, results: list[CollectionResult]) -> int:
        """Ingest validated news articles into the knowledge graph.

        Creates Variable entities for extracted keywords, enabling
        causal analysis of how news topics relate to other variables
        (market prices, prediction market outcomes, etc.).

        Args:
            results: Validated collection results.

        Returns:
            Number of variables created.
        """
        ingested = 0

        for result in results:
            if not result.is_valid:
                continue

            data = result.data
            keywords = data.get("keywords", [])

            for keyword in keywords:
                var_name = f"news:keyword:{keyword.lower()}"
                variable = self._create_variable(
                    name=var_name,
                    variable_type=VariableType.OBSERVABLE,
                )
                result.variables_created.append(variable.id)
                ingested += 1

        logger.info("News: ingested %d keyword variables.", ingested)
        return ingested

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_feed(feed_url: str) -> list[dict[str, Any]]:
        """Parse an RSS feed and return article entries.

        Runs in a thread pool since feedparser is synchronous.

        Args:
            feed_url: URL of the RSS feed.

        Returns:
            List of article dictionaries from feedparser.
        """
        try:
            import feedparser
        except ImportError:
            logger.error(
                "feedparser is not installed. Install it with: pip install feedparser"
            )
            raise ImportError(
                "feedparser is required for the NewsMerchant. "
                "Install with: pip install feedparser"
            )

        feed = feedparser.parse(feed_url)

        if feed.bozo and not feed.entries:
            raise ValueError(f"Failed to parse feed: {feed.bozo_exception}")

        return [
            {
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
                "summary": entry.get("summary", ""),
                "author": entry.get("author", ""),
            }
            for entry in feed.entries
        ]

    @staticmethod
    def _extract_keywords(text: str, max_keywords: int = 10) -> list[str]:
        """Extract keywords from text using simple frequency analysis.

        A lightweight keyword extraction method that tokenizes,
        removes stopwords, and returns the most frequent terms.
        For production use, consider spaCy or a proper NER pipeline.

        Args:
            text: Input text to extract keywords from.
            max_keywords: Maximum number of keywords to return.

        Returns:
            List of extracted keywords, ordered by frequency.
        """
        # Tokenize: split on non-alphanumeric, keep words >= 3 chars
        tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())

        # Remove stopwords
        filtered = [t for t in tokens if t not in _STOPWORDS]

        # Count frequencies
        freq: dict[str, int] = {}
        for token in filtered:
            freq[token] = freq.get(token, 0) + 1

        # Sort by frequency, return top N
        sorted_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [term for term, _count in sorted_terms[:max_keywords]]

    @staticmethod
    def _hash_article(article: dict[str, Any]) -> str:
        """Generate a hash for deduplication of articles.

        Uses the article link and title to create a deterministic hash.

        Args:
            article: Article dictionary with 'link' and 'title' keys.

        Returns:
            Hex digest of the article hash.
        """
        content = f"{article.get('link', '')}:{article.get('title', '')}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def stop(self) -> None:
        """Stop the merchant and shut down the thread pool."""
        self._executor.shutdown(wait=False)
        # Limit seen articles cache size to prevent unbounded memory growth
        if len(self._seen_articles) > 100_000:
            self._seen_articles.clear()
        await super().stop()
