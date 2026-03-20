"""
Merchant agents package — the data-gathering layer of the Republic of AI Agents.

Merchants operate at Pearl's Level 1 (association / observation) for online agents
and Level 2 (intervention) for offline / embodied agents. They follow a standard
pipeline: discover -> collect -> validate -> ingest, feeding data into the
knowledge graph for philosopher-kings to theorize about and warriors to test.

Online merchants pull from digital sources (Polymarket, financial APIs, news,
HuggingFace), while offline merchants ingest data from physical sensors and
embodied robots that interact with the real world.
"""

from .base import CollectionResult, MerchantAgent, MerchantConfig, MerchantStatus
from .registry import MerchantRegistry
from .scheduler import MerchantScheduler

__all__ = [
    # Base abstractions
    "MerchantAgent",
    "MerchantConfig",
    "MerchantStatus",
    "CollectionResult",
    # Registry and scheduling
    "MerchantRegistry",
    "MerchantScheduler",
]
