"""Configuration for the GNN-TCN economic prediction model."""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class EconomicFeatureConfig:
    """12 economic features per city per quarter."""
    step_months: int = 3  # Quarterly
    window_size: int = 16  # 16 quarters = 4 years
    n_features: int = 12
    feature_names: list[str] = field(default_factory=lambda: [
        "gdp_growth",              # F1: GDP growth rate (% YoY)
        "trade_volume_change",     # F2: Trade volume change (% QoQ)
        "fdi_change",              # F3: FDI inflow change (% QoQ)
        "inflation_rate",          # F4: CPI inflation (% annual)
        "exchange_rate_change",    # F5: Exchange rate movement vs USD (% QoQ)
        "port_throughput_change",  # F6: Port/logistics index change
        "avg_tariff_rate",         # F7: Average applied tariff (0-1)
        "trade_balance",           # F8: (exports - imports) / (exports + imports)
        "ease_of_business_change", # F9: Ease of business score delta
        "remittance_change",       # F10: Remittance inflow change (% QoQ)
        "political_stability",     # F11: Political stability index (0-1)
        "regional_trade_share",    # F12: Intra-ECOWAS trade / total trade
    ])


@dataclass
class ModelConfig:
    """GNN-TCN architecture hyperparameters."""
    # GAT
    gat_in_features: int = 12
    gat_hidden: int = 32
    gat_heads: int = 4
    gat_out: int = 32
    gat_dropout: float = 0.1
    # TCN
    tcn_channels: list[int] = field(default_factory=lambda: [64, 64, 64, 64])
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.2
    # Prediction head
    fc_hidden: int = 64
    fc_dropout: float = 0.3
    n_targets: int = 29  # FTZ target cities
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 8  # Smaller batches for quarterly data
    epochs: int = 200
    patience: int = 25
    grad_clip: float = 1.0
    # Platt scaling
    platt_lr: float = 0.01
    platt_epochs: int = 200
    # Window
    window_size: int = 16
    step_months: int = 3


@dataclass
class BacktestConfig:
    """Backtesting parameters for economic predictions."""
    train_end_year: int = 2018
    val_end_year: int = 2021
    test_end_year: int = 2025
    # Metrics
    risk_free_rate: float = 0.05


@dataclass
class GraphConfig:
    """Graph construction for economic similarity."""
    method: str = "combined"
    trade_weight: float = 0.25
    political_weight: float = 0.15
    cultural_weight: float = 0.10
    migration_weight: float = 0.15
    currency_weight: float = 0.15
    infrastructure_weight: float = 0.10
    correlation_weight: float = 0.10
    min_similarity: float = 0.10
    add_self_loops: bool = True
    symmetric: bool = True


@dataclass
class GNNConfig:
    """Top-level config."""
    features: EconomicFeatureConfig = field(default_factory=EconomicFeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    graph_type: str = "geographic"  # 'geographic' or 'economic'
    model_save_dir: str = "gnn/checkpoints"
    log_dir: str = "gnn/logs"
