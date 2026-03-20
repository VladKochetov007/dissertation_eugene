"""Build adjacency matrices from economic similarity between cities."""

from __future__ import annotations

import pathlib
import json
from typing import Optional

import numpy as np

from .config import GraphConfig


DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


class EconomicGraphBuilder:
    """Construct adjacency matrices from economic relationships.

    7 similarity methods, each corresponding to a ConnectionType:
    1. trade -- bilateral trade volume as weight
    2. political -- shared bloc membership
    3. cultural -- shared language, ethnic ties
    4. migration -- migration flow volumes
    5. currency -- same currency zone (CFA)
    6. infrastructure -- physical connectivity
    7. combined -- weighted sum of all above
    """

    def __init__(self, config: Optional[GraphConfig] = None) -> None:
        self.config = config or GraphConfig()

    def build(
        self, data_dir: Optional[pathlib.Path] = None, method: Optional[str] = None
    ) -> np.ndarray:
        """Build adjacency matrix using specified method."""
        d = data_dir or DATA_DIR
        method = method or self.config.method

        with open(d / "cities.json") as f:
            cities = json.load(f)
        with open(d / "edges.json") as f:
            edges = json.load(f)

        city_ids = [c["id"] for c in cities]
        N = len(city_ids)
        city_idx = {cid: i for i, cid in enumerate(city_ids)}
        city_map = {c["id"]: c for c in cities}

        if method == "combined":
            return self._build_combined(cities, edges, city_idx, city_map, N)

        # Single method
        adj = np.zeros((N, N), dtype=np.float32)
        type_map = {
            "trade": "TRADE",
            "political": "POLITICAL",
            "cultural": "CULTURAL",
            "migration": "MIGRATORY",
            "infrastructure": "INFRASTRUCTURE",
            "currency": "FINANCIAL",
        }
        edge_type = type_map.get(method, method.upper())

        for e in edges:
            if e["edge_type"] == edge_type:
                src = city_idx.get(e["source"])
                tgt = city_idx.get(e["target"])
                if src is not None and tgt is not None:
                    w = 1.0 - e.get("weight", 0.5)
                    adj[src, tgt] = max(adj[src, tgt], w)
                    adj[tgt, src] = max(adj[tgt, src], w)

        if self.config.add_self_loops:
            np.fill_diagonal(adj, 1.0)

        # Threshold
        adj[adj < self.config.min_similarity] = 0.0

        # Symmetric normalization
        if self.config.symmetric:
            adj = self._normalize(adj)

        return adj

    def _build_combined(
        self, cities, edges, city_idx, city_map, N
    ) -> np.ndarray:
        """Build combined adjacency from all edge types."""
        # Build per-type matrices
        type_weights = {
            "TRADE": self.config.trade_weight,
            "POLITICAL": self.config.political_weight,
            "CULTURAL": self.config.cultural_weight,
            "MIGRATORY": self.config.migration_weight,
            "FINANCIAL": self.config.currency_weight,
            "INFRASTRUCTURE": self.config.infrastructure_weight,
            "LABOUR": self.config.correlation_weight,
        }

        combined = np.zeros((N, N), dtype=np.float32)

        for edge_type, weight in type_weights.items():
            adj = np.zeros((N, N), dtype=np.float32)
            for e in edges:
                if e["edge_type"] == edge_type:
                    src = city_idx.get(e["source"])
                    tgt = city_idx.get(e["target"])
                    if src is not None and tgt is not None:
                        w = 1.0 - e.get("weight", 0.5)
                        adj[src, tgt] = max(adj[src, tgt], w)
                        adj[tgt, src] = max(adj[tgt, src], w)

            # Normalize per-type matrix to [0, 1]
            max_val = adj.max()
            if max_val > 0:
                adj = adj / max_val

            combined += weight * adj

        if self.config.add_self_loops:
            np.fill_diagonal(combined, 1.0)

        combined[combined < self.config.min_similarity] = 0.0

        if self.config.symmetric:
            combined = self._normalize(combined)

        return combined

    @staticmethod
    def _normalize(adj: np.ndarray) -> np.ndarray:
        """Symmetric normalization: D^{-1/2} A D^{-1/2}."""
        degree = adj.sum(axis=1)
        degree[degree == 0] = 1.0
        d_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
        return d_inv_sqrt @ adj @ d_inv_sqrt
