import pandas as pd


class ConformerAggregator:
    """
    Aggregates conformer-level features to molecule-level features.
    """

    def __init__(self, temperature: float = 300.0) -> None:
        self.temperature = temperature

    def thermal_average(self, df: pd.DataFrame) -> pd.Series:
        ...