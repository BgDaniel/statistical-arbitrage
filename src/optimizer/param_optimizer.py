import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Tuple, List
from tqdm import tqdm
from src.models.pair_arbitrage import PairArbitrage
from src.models.config import ArbitrageConfig
from src.data_handler.data_handler import DataHandler


logger = logging.getLogger(__name__)


class ParamOptimizer:
    """
    Optimize entry/exit z-score thresholds and rolling window length
    for a pair trading strategy.

    Attributes:
        tickers (Tuple[str, str]): Pair of ticker symbols.
        start_date (str): Start date for historical data.
        end_date (str): End date for historical data.
        prices (pd.DataFrame): Historical price data.
        pair_data (dict): Dictionary with aligned price series for the pair.
    """

    def __init__(
        self, tickers: Tuple[str, str], start_date: str, end_date: str
    ) -> None:
        """
        Initialize the ParamOptimizer with tickers and date range.

        Args:
            tickers (Tuple[str, str]): Two ticker symbols for pair trading.
            start_date (str): Start date of the analysis.
            end_date (str): End date of the analysis.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

        logger.info(
            f"Fetching price data for {tickers} from {start_date} to {end_date}..."
        )
        dh = DataHandler(
            tickers=list(tickers), start_date=start_date, end_date=end_date
        )
        self.prices = dh.fetch_data()
        logger.info("Price data fetched successfully.")

        self.pair_data = {
            tickers[0]: self.prices[tickers[0]],
            tickers[1]: self.prices[tickers[1]],
        }

    def optimize_params(
        self,
        entry_range: np.ndarray,
        exit_range: np.ndarray,
        window_range: List[int],
        capital: float = 1000.0,
        max_long: float = 1000.0,
        max_short: float = 1000.0,
        plot_heatmaps: bool = False,
    ) -> dict:
        """
        Backtest over all valid combinations of entry_z, exit_z, and rolling window.

        Args:
            entry_range (np.ndarray): Array of entry z-score thresholds.
            exit_range (np.ndarray): Array of exit z-score thresholds.
            window_range (List[int]): List of rolling window lengths.
            capital (float): Starting capital.
            max_long (float): Max capital allowed in long positions.
            max_short (float): Max capital allowed in short positions.
            plot_heatmaps (bool): Whether to plot heatmaps for each window length.

        Returns:
            dict: Dictionary with window length as keys and pd.DataFrame of returns as values.
        """
        results_dict = {}
        optimal_params = {
            "window": None,
            "entry_z": None,
            "exit_z": None,
            "return": -np.inf,
        }

        for window in window_range:
            logger.info(f"Testing window length: {window}")
            config = ArbitrageConfig(window=window, plot=False)
            arb = PairArbitrage(self.pair_data, config)
            arb.analyze(plot=False)

            results = pd.DataFrame(index=exit_range, columns=entry_range, dtype=float)

            total_combinations = sum(
                1
                for exit_z in exit_range
                for entry_z in entry_range
                if exit_z <= entry_z
            )
            logger.info(f"Total combinations for this window: {total_combinations}")

            with tqdm(
                total=total_combinations, desc=f"Window {window}", ncols=100
            ) as pbar:
                for exit_z in exit_range:
                    for entry_z in entry_range:
                        if exit_z > entry_z:
                            continue
                        df = arb.backtest_strategy(
                            entry_z=entry_z,
                            exit_z=exit_z,
                            capital=capital,
                            max_long=max_long,
                            max_short=max_short,
                            plot=False,
                        )
                        total_ret = df["total_return"].iloc[-1]
                        results.at[exit_z, entry_z] = total_ret

                        if total_ret > optimal_params["return"]:
                            optimal_params.update(
                                {
                                    "window": window,
                                    "entry_z": entry_z,
                                    "exit_z": exit_z,
                                    "return": total_ret,
                                }
                            )
                        pbar.update(1)

            results_dict[window] = results

        if plot_heatmaps:
            n_windows = len(window_range)
            n_cols = min(3, n_windows)  # max 3 columns
            n_rows = int(np.ceil(n_windows / n_cols))

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)

            for i, window in enumerate(window_range):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col]
                df = results_dict[window]

                im = ax.imshow(df.values.astype(float), origin="lower", aspect="auto", cmap="RdYlGn")

                ax.set_xticks(np.arange(len(entry_range)))
                ax.set_xticklabels([f"{v:.2f}" for v in entry_range])
                ax.set_yticks(np.arange(len(exit_range)))
                ax.set_yticklabels([f"{v:.2f}" for v in exit_range])
                ax.set_xlabel("Entry Z")
                ax.set_ylabel("Exit Z")
                ax.set_title(f"Window={window}")

                max_idx = np.unravel_index(np.nanargmax(df.values), df.shape)
                ax.plot(max_idx[1], max_idx[0], "x", color="black", markersize=12, mew=3)

                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label("Total Return")

            # Hide any unused subplots
            for j in range(i + 1, n_rows * n_cols):
                fig.delaxes(axes.flatten()[j])

            plt.tight_layout()
            plt.show()

        logger.info("Optimal Parameters Found:")
        logger.info(f"Window Length: {optimal_params['window']}")
        logger.info(f"Entry Z: {optimal_params['entry_z']}")
        logger.info(f"Exit Z: {optimal_params['exit_z']}")
        logger.info(f"Total Return: {optimal_params['return']:.2%}")

        return optimal_params, results_dict


if __name__ == "__main__":
    tickers = ("SPY", "IVV")
    start_date = "2020-01-01"
    end_date = "2025-11-01"

    entry_range = np.arange(0.1, 6.1, 0.5)
    exit_range = np.arange(0.05, 4.1, 0.5)
    window_range = [5, 10, 15, 30, 60, 90]

    optimizer = ParamOptimizer(tickers, start_date, end_date)
    optimal_params, results = optimizer.optimize_params(
        entry_range, exit_range, window_range, plot_heatmaps=True
    )
