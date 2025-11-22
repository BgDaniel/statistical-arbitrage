from dataclasses import dataclass

@dataclass
class ArbitrageConfig:
    """
    Configuration object for the PairArbitrage class.
    """
    window: int = 120
    plot: bool = True
