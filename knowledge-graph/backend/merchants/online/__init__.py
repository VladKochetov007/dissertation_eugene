"""
Online merchant agents — Pearl's Level 1 (association / observation).

These agents gather data from digital sources: prediction markets, financial
APIs, news feeds, and ML dataset registries. They observe and record but do
not intervene — they watch the world and report what correlates with what.

Available merchants:
- PolymarketMerchant: prediction market prices, trades, and event metadata
- FinancialMerchant: stock prices, forex rates, commodity data via yfinance
- NewsMerchant: RSS feed aggregation with keyword/entity extraction
- HuggingFaceMerchant: dataset discovery and model metadata from HuggingFace Hub
"""

from .financial import FinancialMerchant
from .huggingface import HuggingFaceMerchant
from .news import NewsMerchant
from .polymarket import PolymarketMerchant

__all__ = [
    "PolymarketMerchant",
    "FinancialMerchant",
    "NewsMerchant",
    "HuggingFaceMerchant",
]
