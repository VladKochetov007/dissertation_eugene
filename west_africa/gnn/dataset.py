"""Dataset for the West Africa GNN-TCN model."""

from __future__ import annotations

import json
import pathlib
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import EconomicFeatureConfig


DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


class WestAfricaTemporalDataset(Dataset):
    """Temporal dataset for GNN-TCN training on economic indicators.

    Each sample: (node_features, adjacency, label)
    - node_features: (N_cities, window_size, n_features)
    - adjacency: (N_cities, N_cities)
    - label: (n_targets,) -- next-quarter trade volume change for target cities
    """

    def __init__(
        self,
        features: np.ndarray,        # (T, N, F) time-series of city features
        adjacency: np.ndarray,        # (N, N) adjacency matrix
        target_indices: list[int],    # indices of FTZ target cities in node list
        config: Optional[EconomicFeatureConfig] = None,
        stride: int = 1,              # stride between samples (in quarters)
    ):
        self.config = config or EconomicFeatureConfig()
        self.features = features
        self.adjacency = adjacency
        self.target_indices = target_indices
        self.stride = stride
        self.window = self.config.window_size

        T = features.shape[0]
        self.n_samples = max(0, (T - self.window - 1) // stride)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        end = start + self.window

        # Node features: (N, W, F)
        x = self.features[start:end].transpose(1, 0, 2)  # (T, N, F) -> (N, T, F)
        x = torch.tensor(x, dtype=torch.float32)

        # Adjacency
        adj = torch.tensor(self.adjacency, dtype=torch.float32)

        # Label: next-quarter values for target nodes
        label_step = end  # next quarter after window
        if label_step < self.features.shape[0]:
            # Use trade_volume_change (feature index 1) as prediction target
            label = self.features[label_step, self.target_indices, 1]
        else:
            label = np.zeros(len(self.target_indices))
        label = torch.tensor(label, dtype=torch.float32)

        return x, adj, label

    @classmethod
    def from_seed_data(
        cls,
        data_dir: Optional[pathlib.Path] = None,
        config: Optional[EconomicFeatureConfig] = None,
    ) -> "WestAfricaTemporalDataset":
        """Build dataset from seed JSON files (for testing/demo purposes).

        Uses economic_state.json to create a synthetic time series.
        """
        d = data_dir or DATA_DIR
        config = config or EconomicFeatureConfig()

        with open(d / "cities.json") as f:
            cities = json.load(f)
        with open(d / "edges.json") as f:
            edges = json.load(f)
        with open(d / "economic_state.json") as f:
            states = json.load(f)

        city_ids = [c["id"] for c in cities]
        N = len(city_ids)
        city_idx = {cid: i for i, cid in enumerate(city_ids)}

        # Build adjacency matrix
        adj = np.eye(N, dtype=np.float32)  # self-loops
        for e in edges:
            src = city_idx.get(e["source"])
            tgt = city_idx.get(e["target"])
            if src is not None and tgt is not None:
                w = 1.0 - e.get("weight", 0.5)  # Convert: lower weight = stronger connection
                adj[src, tgt] = max(adj[src, tgt], w)
                adj[tgt, src] = max(adj[tgt, src], w)

        # Build synthetic time series from economic states
        state_map = {s["city_id"]: s for s in states}
        T = config.window_size + 10  # enough for a few samples

        features = np.zeros((T, N, config.n_features), dtype=np.float32)
        for i, cid in enumerate(city_ids):
            st = state_map.get(cid, {})
            base = np.array([
                st.get("gdp_growth_rate", 0.0),
                st.get("trade_volume_change", 0.0),
                st.get("fdi_inflow_change", 0.0),
                st.get("inflation_rate", 0.0),
                0.0,  # exchange_rate_change
                0.0,  # port_throughput_change
                st.get("tariff_change", 0.0),
                0.0,  # trade_balance
                0.0,  # ease_of_business_change
                0.0,  # remittance_change
                st.get("political_stability", 0.5),
                0.0,  # regional_trade_share
            ], dtype=np.float32)
            # Add small noise over time to create variation
            rng = np.random.RandomState(hash(cid) % 2**31)
            for t in range(T):
                noise = rng.randn(config.n_features).astype(np.float32) * 0.05
                features[t, i] = base + noise * (t + 1) * 0.01

        # Target indices: FTZ target cities
        target_indices = [
            city_idx[c["id"]] for c in cities if c.get("is_ftz_target", False)
        ]

        return cls(features, adj, target_indices, config)
