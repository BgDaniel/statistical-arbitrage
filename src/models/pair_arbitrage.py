import numpy as np
import pandas as pd
from typing import Dict, Any
from statsmodels.tsa.stattools import adfuller
import logging
from scipy.stats import shapiro, jarque_bera, norm, probplot
import matplotlib.pyplot as plt

from src.models.kalman_filter import KalmanFilter
from src.models.config import ArbitrageConfig


logger = logging.getLogger(__name__)


class PairArbitrage:
    """
    A class that performs statistical arbitrage analysis on two price series.

    Inputs:
        - data: Dict[str, pd.Series] with exactly two tickers
        - config: ArbitrageConfig

    The class:
        1. Estimates time-varying beta using a Kalman filter.
        2. Computes the spread = y - beta*x.
        3. Performs a rolling ADF test.
        4. Optionally plots beta(t) and spread(t).
    """

    def __init__(self, data: Dict[str, pd.Series], config: ArbitrageConfig) -> None:
        if len(data) != 2:
            raise ValueError("PairArbitrage requires exactly two price series.")

        self.tickers = list(data.keys())
        self.y = data[self.tickers[0]].dropna()
        self.x = data[self.tickers[1]].dropna()
        self.config = config

        # Align time index
        df = pd.concat([self.y, self.x], axis=1).dropna()
        self.y = df[self.tickers[0]]
        self.x = df[self.tickers[1]]

    def _estimate_beta_kalman(self) -> pd.Series:
        """
        Estimate time-varying beta using a general Kalman filter.
        """

        F = np.array([[1.0]])  # Beta follows random walk
        Q = np.array([[1e-5]])
        R = np.array([[1e-2]])

        # H_t changes every time step (depends on x_t)
        betas = []

        kf = KalmanFilter(F=F, H=np.array([[0.0]]), Q=Q, R=R)

        for x_t, y_t in zip(self.x.values, self.y.values):
            H_t = np.array([[x_t]])
            kf.H = H_t
            state, _ = kf.step(np.array([y_t]))
            betas.append(state.item())

        return pd.Series(betas, index=self.y.index)

    def _compute_spread(self, beta: pd.Series) -> pd.Series:
        """
        Compute spread process.
        """
        return self.y - beta * self.x

    def _diagnose_spread(self, spread: pd.Series, plot: bool = True) -> Dict[str, Any]:
        """
        Diagnose spread properties:
            - Normality tests: Shapiro-Wilk, Jarque-Bera
            - Stationarity: ADF
            - Optionally: plot empirical distribution + fitted normal + QQ-plot

        Returns:
            dict containing test results
        """

        # --- Drop NaN ---
        s = spread.dropna()

        # --- Stationarity (ADF) ---
        try:
            adf_stat, adf_pval, _, _, _, _ = adfuller(s, autolag="AIC")
        except Exception as e:
            logger.warning(f"ADF failed: {e}")
            adf_stat, adf_pval = np.nan, np.nan

        # --- Normality tests ---
        try:
            sh_stat, sh_pval = shapiro(s)
        except Exception as e:
            logger.warning(f"Shapiro test failed: {e}")
            sh_stat, sh_pval = np.nan, np.nan

        try:
            jb_stat, jb_pval = jarque_bera(s)
        except Exception as e:
            logger.warning(f"Jarque-Bera test failed: {e}")
            jb_stat, jb_pval = np.nan, np.nan

        results = {
            "adf_stat": adf_stat,
            "adf_pval": adf_pval,
            "shapiro_stat": sh_stat,
            "shapiro_pval": sh_pval,
            "jb_stat": jb_stat,
            "jb_pval": jb_pval,
        }

        # --- Plot: distribution + QQ plot ---
        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # ---- LEFT: empirical + fitted normal ----
            ax = axes[0]
            ax.hist(
                s, bins=40, density=True, alpha=0.6, color="gray", label="Empirical"
            )

            mu, sigma = s.mean(), s.std()
            xs = np.linspace(s.min(), s.max(), 200)
            ax.plot(xs, norm.pdf(xs, mu, sigma), color="red", label="Normal PDF")

            ax.set_title("Empirical vs Normal PDF")
            ax.grid(True)
            ax.legend()

            # ---- RIGHT: QQ plot ----
            ax = axes[1]
            probplot(s, dist="norm", plot=ax)
            ax.set_title("QQ Plot: Spread")
            ax.grid(True)

            plt.tight_layout()
            plt.show()

        return results

    def analyze(self, plot: bool = True) -> Dict[str, Any]:
        """
        Run full analysis:
            - Kalman Beta
            - Spread
            - Rolling ADF
            - Plotting
        Stores beta and spread in instance variables to reuse in backtest.
        """

        self.beta = self._estimate_beta_kalman()
        self.spread = self._compute_spread(self.beta)
        diag = self._diagnose_spread(self.spread, plot=False)

        return {
            "beta": self.beta,
            "spread": self.spread,
            "diagnostics": diag,
        }

    def _compute_capital(
            self,
            position: pd.Series,
            initial_capital: float,
            max_long: float,
            max_short: float
    ) -> pd.DataFrame:
        """
        Compute daily total capital considering all position transitions,
        limiting the exposure in long and short positions by absolute leg amounts.
        Tracks capital invested in each underlying asset.

        Args:
            position: Series of positions (-1, 0, +1)
            prev_capital: Starting capital
            max_long: Max € allowed to invest in x when long
            max_short: Max € allowed to invest in y when short

        Returns:
            pd.DataFrame: columns ["total_capital", "pos_x", "pos_y"]
        """
        capital_series = [initial_capital]

        for t in range(1, len(self.spread)):
            prev_dir = position.iloc[t - 1]
            curr_dir = position.iloc[t]
            prev_capital = capital_series[-1]

            prev_spread = self.spread.iloc[t - 1]
            curr_spread = self.spread.iloc[t]

            prev_x_price = self.x.iloc[t - 1]
            prev_y_price = self.y.iloc[t - 1]

            prev_beta_t = self.beta.iloc[t - 1]

            prev_delta = prev_dir * prev_capital / abs(prev_spread)

            # For x leg (scaled by beta)
            if -prev_delta * prev_beta_t * prev_x_price > max_long:
                prev_delta = -max_long / (prev_beta_t * prev_x_price)
            elif -prev_delta * prev_beta_t * prev_x_price < -max_short:
                prev_delta = max_short / (prev_beta_t * prev_x_price)

            # For y leg
            if prev_delta * prev_y_price > max_long:
                prev_delta = max_long / prev_y_price
            elif prev_delta * prev_y_price < -max_short:
                prev_delta = -max_short / prev_y_price

            delta_capital = prev_delta * (curr_spread - prev_spread)

            if prev_dir != 0:
                prev_capital += delta_capital

            # Prevent negative capital
            prev_capital = max(prev_capital, 0.0)

            capital_series.append(prev_capital)

        return pd.Series(capital_series, index=self.spread.index, name="total_capital")

    def backtest_strategy(
        self,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        capital: float = 1000.0,
        max_long: float = 10000.0,
        max_short: float = 10000.0,
        plot: bool = True,
    ) -> pd.DataFrame:
        """
        Backtest a mean-reversion statistical arbitrage strategy using full reinvestment.

        Args:
            entry_z (float): Entry threshold for z-score.
            exit_z (float): Exit threshold for z-score.
            capital (float): Initial investment capital.
            max_long_eur: Maximum € allowed to invest in long spread
            max_short_eur: Maximum € allowed to invest in short spread
            plot (bool): Whether to plot results.

        Returns:
            pd.DataFrame: Contains spread, zscore, position, daily PnL, cumulative PnL, capital, overall return.
        """
        spread = self.spread
        window = self.config.window

        # --- Rolling stats ---
        roll_mean = spread.rolling(window).mean()
        roll_std = spread.rolling(window).std()
        zscore = (spread - roll_mean) / roll_std

        # --- Trading signals ---
        position = np.zeros(len(spread))
        for t in range(1, len(spread)):
            prev_pos = position[t - 1]
            z = zscore.iloc[t]

            if prev_pos == 0:  # currently flat
                if z > entry_z:
                    position[t] = -1  # enter short
                elif z < -entry_z:
                    position[t] = 1  # enter long
                else:
                    position[t] = 0  # stay flat
            elif prev_pos == 1:  # currently long
                if z >= -exit_z:
                    position[t] = 0  # exit long
                else:
                    position[t] = 1  # hold long
            elif prev_pos == -1:  # currently short
                if z <= exit_z:
                    position[t] = 0  # hold short
                else:
                    position[t] = -1  # exit short

        position = pd.Series(position, index=spread.index)

        # --- Compute daily total capital using the helper function ---
        capital_series = self._compute_capital(
            position, capital, max_long, max_short
        )

        # Compute overall return
        total_return = (capital_series.iloc[-1] - capital) / capital

        # Compile result dataframe
        result = pd.DataFrame(
            {
                "spread": spread,
                "zscore": zscore,
                "position": position,
                "total_capital": capital_series,
                "total_return": total_return,
            }
        )

        # --- Plotting ---
        if plot and self.config.plot:
            fig, axs = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

            # 1. Beta timeseries (if available)
            if hasattr(self, "beta"):
                axs[0].plot(self.beta.index, self.beta, color="blue", label="Beta")
                axs[0].set_title("Beta Time Series")
                axs[0].grid(True)
                axs[0].legend()

            # 2a. Spread
            axs[1].plot(spread.index, spread, color="green", label="Spread")
            axs[1].set_title("Spread")
            axs[1].set_ylabel("Spread")
            axs[1].grid(True)
            axs[1].legend()

            # 2a. Spread
            axs[1].plot(spread.index, spread, color="green", label="Spread")
            axs[1].set_title("Spread")
            axs[1].set_ylabel("Spread")
            axs[1].grid(True)
            axs[1].legend()

            # 2b. Z-score with entry/exit thresholds and LONG/SHORT markers
            axs[2].plot(
                zscore.index, zscore, color="grey", label="Z-score", linewidth=1
            )
            axs[2].axhline(
                entry_z, color="red", linestyle="--", label="Entry Threshold"
            )
            axs[2].axhline(-entry_z, color="red", linestyle="--")
            axs[2].axhline(exit_z, color="blue", linestyle="--", label="Exit Threshold")
            axs[2].axhline(-exit_z, color="blue", linestyle="--")

            # Scatter LONG/SHORT signals
            buy_idx = result.index[result["position"] == 1]
            sell_idx = result.index[result["position"] == -1]
            axs[2].scatter(
                buy_idx,
                zscore.loc[buy_idx],
                color="blue",
                marker="^",
                s=60,
                label="LONG",
            )
            axs[2].scatter(
                sell_idx,
                zscore.loc[sell_idx],
                color="red",
                marker="v",
                s=60,
                label="SHORT",
            )

            axs[2].set_title("Z-score with Entry/Exit Levels")
            axs[2].set_ylabel("Z-score")
            axs[2].grid(True)
            axs[2].legend(loc="upper right")

            # 3. Total capital
            axs[3].plot(
                capital_series.index,
                capital_series,
                color="purple",
                linewidth=2,
                label="Total Capital",
            )
            axs[3].set_title(
                f"Daily Total Capital Curve (Total Return: {total_return:.2%})"
            )
            axs[3].set_xlabel("Date")
            axs[3].set_ylabel("Capital")
            axs[3].grid(True)
            axs[3].legend()

            plt.tight_layout()
            plt.show()

        return result
