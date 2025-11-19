from dataclasses import dataclass

@dataclass
class ArbitrageConfig:
    """
    Configuration object for the PairArbitrage class.
    """
    window: int = 120
    adf_threshold: float = 0.05
    plot: bool = True
