"""
HuggingFace Merchant Agent — dataset discovery and model metadata collection.

Integrates with the HuggingFace Hub to discover datasets and models relevant
to the knowledge graph's hypotheses. HuggingFace is the Republic of Letters
for ML: a commons where models and data are shared, evaluated, and built upon.

Data collected:
- Dataset metadata (name, description, tags, size, downloads)
- Model card information (architecture, training data, metrics)
- Task categories and benchmark results

Pearl Level: 1 (Association) — observational metadata about ML artifacts.

Usage:
    from graph.store import KnowledgeGraphStore
    from merchants.online.huggingface import HuggingFaceMerchant

    store = KnowledgeGraphStore()
    merchant = HuggingFaceMerchant(
        store=store,
        search_queries=["causal inference", "prediction markets", "time series"],
    )
    await merchant.start()
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

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

HF_API_BASE = "https://huggingface.co/api"
DEFAULT_POLL_INTERVAL = 3600.0  # 1 hour (HF data doesn't change rapidly)
DEFAULT_RATE_LIMIT = 10.0  # requests per minute

DEFAULT_SEARCH_QUERIES = [
    "causal inference",
    "prediction markets",
    "time series forecasting",
    "financial data",
    "news classification",
]


# ---------------------------------------------------------------------------
# HuggingFace Merchant
# ---------------------------------------------------------------------------


class HuggingFaceMerchant(MerchantAgent):
    """Merchant agent for HuggingFace Hub dataset and model discovery.

    Searches the HuggingFace Hub for datasets and models matching
    configured queries, collecting metadata and linking ML artifacts
    to the knowledge graph as potential data sources for hypothesis testing.

    Attributes:
        search_queries: List of search terms for dataset/model discovery.
        include_models: Whether to search for models in addition to datasets.
        max_results_per_query: Maximum results per search query.
        api_token: Optional HuggingFace API token for authenticated requests.
        _client: Async HTTP client.
        _discovered_datasets: Cache of already-discovered dataset IDs.
    """

    def __init__(
        self,
        store: KnowledgeGraphStore,
        search_queries: Optional[list[str]] = None,
        include_models: bool = False,
        max_results_per_query: int = 20,
        api_token: Optional[str] = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        rate_limit: float = DEFAULT_RATE_LIMIT,
    ) -> None:
        """Initialize the HuggingFace merchant.

        Args:
            store: Knowledge graph store for data ingestion.
            search_queries: Search terms for discovery. Defaults to
                           causal-inference-related queries.
            include_models: Whether to also search for models.
            max_results_per_query: Max results returned per query.
            api_token: HuggingFace API token for higher rate limits.
            poll_interval: Seconds between polling cycles.
            rate_limit: Maximum requests per minute.
        """
        config = MerchantConfig(
            name="huggingface-hub",
            source_type=DataSourceType.DATASET,
            pearl_level=PearlLevel.ASSOCIATION,
            poll_interval_seconds=poll_interval,
            rate_limit_requests_per_minute=rate_limit,
            metadata={
                "include_models": include_models,
                "max_results_per_query": max_results_per_query,
            },
        )
        super().__init__(config=config, store=store)

        self.search_queries = (
            search_queries if search_queries is not None else list(DEFAULT_SEARCH_QUERIES)
        )
        self.include_models = include_models
        self.max_results_per_query = max_results_per_query
        self.api_token = api_token
        self._client: Optional[httpx.AsyncClient] = None
        self._discovered_datasets: set[str] = set()

        # Register data source
        self._data_source = self._register_data_source(
            name="HuggingFace Hub",
            url="https://huggingface.co",
            schema_info={
                "type": "ml_registry",
                "provider": "huggingface",
                "data_fields": [
                    "dataset_id",
                    "description",
                    "tags",
                    "downloads",
                    "likes",
                    "task_categories",
                    "size_category",
                ],
            },
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client.

        Returns:
            Configured httpx.AsyncClient with optional auth headers.
        """
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {"Accept": "application/json"}
            if self.api_token:
                headers["Authorization"] = f"Bearer {self.api_token}"

            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                headers=headers,
            )
        return self._client

    # ------------------------------------------------------------------
    # Pipeline implementation
    # ------------------------------------------------------------------

    async def discover(self) -> list[str]:
        """Discover datasets on HuggingFace Hub matching search queries.

        Queries the HuggingFace API for datasets matching each configured
        search term, returning dataset IDs for detailed collection.

        Returns:
            List of HuggingFace dataset identifiers (e.g., "username/dataset-name").
        """
        client = await self._get_client()
        discovered: list[str] = []

        for query in self.search_queries:
            try:
                await self._enforce_rate_limit()

                response = await client.get(
                    f"{HF_API_BASE}/datasets",
                    params={
                        "search": query,
                        "limit": self.max_results_per_query,
                        "sort": "downloads",
                        "direction": -1,
                    },
                )
                response.raise_for_status()
                datasets = response.json()

                for ds in datasets:
                    ds_id = ds.get("id", "")
                    if ds_id and ds_id not in self._discovered_datasets:
                        discovered.append(ds_id)
                        self._discovered_datasets.add(ds_id)

            except httpx.HTTPError as exc:
                logger.warning(
                    "HuggingFace: discovery failed for query '%s': %s",
                    query,
                    exc,
                )

        # Optionally discover models too
        if self.include_models:
            for query in self.search_queries:
                try:
                    await self._enforce_rate_limit()

                    response = await client.get(
                        f"{HF_API_BASE}/models",
                        params={
                            "search": query,
                            "limit": self.max_results_per_query,
                            "sort": "downloads",
                            "direction": -1,
                        },
                    )
                    response.raise_for_status()
                    models = response.json()

                    for model in models:
                        model_id = model.get("id", "")
                        if model_id:
                            discovered.append(f"model:{model_id}")

                except httpx.HTTPError as exc:
                    logger.warning(
                        "HuggingFace: model discovery failed for query '%s': %s",
                        query,
                        exc,
                    )

        logger.info("HuggingFace: discovered %d new artifacts.", len(discovered))
        return discovered

    async def collect(self, targets: list[str]) -> list[CollectionResult]:
        """Collect detailed metadata for each discovered dataset or model.

        Fetches dataset cards, download counts, tags, and other metadata
        from the HuggingFace API.

        Args:
            targets: List of dataset/model identifiers.

        Returns:
            List of CollectionResult objects with metadata.
        """
        client = await self._get_client()
        results: list[CollectionResult] = []

        for target_id in targets:
            try:
                await self._enforce_rate_limit()

                # Determine if this is a model or dataset
                is_model = target_id.startswith("model:")
                actual_id = target_id.removeprefix("model:") if is_model else target_id
                endpoint = "models" if is_model else "datasets"

                response = await client.get(
                    f"{HF_API_BASE}/{endpoint}/{actual_id}",
                )

                if response.status_code == 404:
                    logger.debug("HuggingFace: %s not found, skipping.", target_id)
                    continue

                response.raise_for_status()
                metadata = response.json()

                result = CollectionResult(
                    target=target_id,
                    data={
                        "id": actual_id,
                        "type": "model" if is_model else "dataset",
                        "description": metadata.get("description", ""),
                        "tags": metadata.get("tags", []),
                        "downloads": metadata.get("downloads", 0),
                        "likes": metadata.get("likes", 0),
                        "created_at": metadata.get("createdAt"),
                        "last_modified": metadata.get("lastModified"),
                        "task_categories": metadata.get("taskCategories", []),
                        "size_category": metadata.get("sizeCategory"),
                        "card_data": metadata.get("cardData", {}),
                        "author": metadata.get("author", ""),
                    },
                    provenance=self._build_provenance(
                        source_url=f"https://huggingface.co/{endpoint}/{actual_id}",
                        collection_method="api_poll",
                        confidence=0.9,
                        raw_record_count=1,
                    ),
                )
                results.append(result)

            except httpx.HTTPError as exc:
                logger.warning(
                    "HuggingFace: failed to collect %s: %s", target_id, exc
                )
                results.append(
                    CollectionResult(
                        target=target_id,
                        data={},
                        provenance=self._build_provenance(
                            source_url=f"{HF_API_BASE}/datasets/{target_id}",
                            confidence=0.0,
                        ),
                        is_valid=False,
                        validation_errors=[f"HTTP error: {exc}"],
                    )
                )

        logger.info(
            "HuggingFace: collected metadata for %d/%d artifacts.",
            len([r for r in results if not r.validation_errors]),
            len(targets),
        )
        return results

    async def validate(self, results: list[CollectionResult]) -> list[CollectionResult]:
        """Validate collected HuggingFace metadata.

        Checks for required fields and reasonable values.

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

            if not data.get("id"):
                errors.append("Missing artifact ID.")

            artifact_type = data.get("type")
            if artifact_type not in ("dataset", "model"):
                errors.append(f"Invalid artifact type: {artifact_type}")

            # Downloads should be non-negative
            downloads = data.get("downloads", 0)
            if isinstance(downloads, (int, float)) and downloads < 0:
                errors.append(f"Negative download count: {downloads}")

            result.validation_errors = errors
            result.is_valid = len(errors) == 0

        valid_count = sum(1 for r in results if r.is_valid)
        logger.info(
            "HuggingFace: validated %d/%d results.", valid_count, len(results)
        )
        return results

    async def ingest(self, results: list[CollectionResult]) -> int:
        """Ingest validated HuggingFace metadata into the knowledge graph.

        Creates Variable entities representing datasets and models,
        linking them as potential data sources for hypothesis testing.

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
            artifact_id = data["id"]
            artifact_type = data["type"]

            # Create a variable for the artifact
            var_name = f"hf:{artifact_type}:{artifact_id}"
            variable = self._create_variable(
                name=var_name,
                variable_type=VariableType.OBSERVABLE,
            )
            result.variables_created.append(variable.id)
            ingested += 1

            # Create variables for each tag (enables tag-based queries)
            tags = data.get("tags", [])
            for tag in tags[:5]:  # Limit to top 5 tags
                tag_var_name = f"hf:tag:{tag}"
                tag_var = self._create_variable(
                    name=tag_var_name,
                    variable_type=VariableType.OBSERVABLE,
                )
                result.variables_created.append(tag_var.id)
                ingested += 1

        logger.info("HuggingFace: ingested %d variables.", ingested)
        return ingested

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def stop(self) -> None:
        """Stop the merchant and close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        await super().stop()
