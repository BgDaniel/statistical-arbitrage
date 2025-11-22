import numpy as np
import pandas as pd
import logging
from typing import List, Tuple

from src.optimizer.param_optimizer import ParamOptimizer


logger = logging.getLogger(__name__)


class BatchParamOptimizer:
    """
    Run ParamOptimizer over multiple ticker pairs and store optimal parameters.

    Attributes:
        ticker_pairs (List[Tuple[str,str]]): List of ticker symbol pairs.
        start_date (str): Start date for analysis.
        end_date (str): End date for analysis.
        optimal_df (pd.DataFrame): DataFrame storing optimal parameters for each pair.
    """

    def __init__(
        self, ticker_pairs: List[Tuple[str, str]], start_date: str, end_date: str
    ) -> None:

        self.ticker_pairs = ticker_pairs
        self.start_date = start_date
        self.end_date = end_date

        self.optimal_df = pd.DataFrame(
            columns=["Pair", "Window", "Entry Z", "Exit Z", "Return"]
        )

    def run_optimization(
        self,
        entry_range: np.ndarray,
        exit_range: np.ndarray,
        window_range: List[int],
        capital: float = 1000.0,
        max_long: float = 1000.0,
        max_short: float = 1000.0,
        plot_heatmaps: bool = False,
    ) -> pd.DataFrame:
        """
        Execute ParamOptimizer for each ticker pair and store optimal parameters.

        Returns:
            pd.DataFrame: Summary table of optimal parameters for all pairs.
        """

        results = []

        for tickers in self.ticker_pairs:
            logger.info(f"Optimizing pair: {tickers}")

            optimizer = ParamOptimizer(tickers, self.start_date, self.end_date)

            # Expect optimize_params to return both results_dict + optimal_params
            optimal_params, _ = optimizer.optimize_params(
                entry_range,
                exit_range,
                window_range,
                capital=capital,
                max_long=max_long,
                max_short=max_short,
                plot_heatmaps=plot_heatmaps,
            )

            results.append(
                {
                    "Pair": f"{tickers[0]}-{tickers[1]}",
                    "Window": optimal_params["window"],
                    "Entry Z": optimal_params["entry_z"],
                    "Exit Z": optimal_params["exit_z"],
                    "Return": f"{optimal_params['return'] * 100:.2f}%"
                }
            )

        self.optimal_df = pd.DataFrame(results)

        print("\n===== Optimal Parameters Summary for All Pairs =====")
        print(self.optimal_df.to_string(index=False))

        return self.optimal_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ticker_pairs = [("AAPL", "MSFT"), ("GOOG", "AMZN")]
    start_date = "2020-01-01"
    end_date = "2025-11-01"

    entry_range = np.arange(1.0, 4.1, 0.5)
    exit_range = np.arange(0.5, 3.1, 0.5)
    window_range = [5, 10, 15, 30, 60, 90]

    batch_opt = BatchParamOptimizer(ticker_pairs, start_date, end_date)

    df = batch_opt.run_optimization(
        entry_range, exit_range, window_range, plot_heatmaps=False
    )
