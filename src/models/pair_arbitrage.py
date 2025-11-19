import numpy as np
import pandas as pd
from typing import Dict, Any
from statsmodels.tsa.stattools import adfuller
import logging
from scipy.stats import shapiro, jarque_bera, norm, probplot
import matplotlib.pyplot as plt

from src.models.kalman_filter import KalmanFilter
from src.models.config import ArbitrageConfig
from src.plots.plot_utils import plot_beta_and_spread


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
        diag = self._diagnose_spread(self.spread, plot=plot)

        # Plotting
        if plot and self.config.plot:
            plot_beta_and_spread(self.beta, self.spread)

        return {
            "beta": self.beta,
            "spread": self.spread,
            "diagnostics": diag,
        }

    @staticmethod
    def _compute_pnl_with_reinvestment(
        spread: pd.Series, position: pd.Series, initial_capital: float
    ) -> pd.DataFrame:
        """
        Compute daily and cumulative PnL assuming full reinvestment.

        Args:
            spread: spread series
            position: position series (-1, 0, +1)
            initial_capital: starting capital

        Returns:
            DataFrame with daily PnL, cumulative PnL, capital over time
        """
        capital = initial_capital
        pnl_list = []
        cum_pnl_list = []
        capital_list = []

        for t in range(len(spread)):
            if t == 0:
                pnl = 0.0
            else:
                pnl = (
                    position.iloc[t - 1]
                    * (spread.iloc[t] - spread.iloc[t - 1])
                    * capital
                    / spread.iloc[t - 1]
                )
            capital += pnl
            pnl_list.append(pnl)
            cum_pnl_list.append(capital - initial_capital)
            capital_list.append(capital)

        return pd.DataFrame(
            {"pnl": pnl_list, "cum_pnl": cum_pnl_list, "capital": capital_list},
            index=spread.index,
        )

    def backtest_strategy(
        self,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        capital: float = 1000.0,
        plot: bool = True,
    ) -> pd.DataFrame:
        """
        Backtest a mean-reversion statistical arbitrage strategy using full reinvestment.

        Args:
            entry_z (float): Entry threshold for z-score.
            exit_z (float): Exit threshold for z-score.
            capital (float): Initial investment capital.
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
            if zscore.iloc[t] > entry_z:
                position[t] = -1  # short spread
            elif zscore.iloc[t] < -entry_z:
                position[t] = +1  # long spread
            elif abs(zscore.iloc[t]) < exit_z:
                position[t] = 0  # flat
            else:
                position[t] = position[t - 1]  # hold position

        position = pd.Series(position, index=spread.index)

        # --- Compute PnL using the helper function ---
        pnl_df = PairArbitrage._compute_pnl_with_reinvestment(spread, position, capital)
        daily_pnl = pnl_df["pnl"]
        cum_pnl = pnl_df["cum_pnl"]
        capital_series = pnl_df["capital"]
        total_return = (capital_series.iloc[-1] - capital) / capital

        # --- Annual returns ---
        if isinstance(spread.index, pd.DatetimeIndex):
            df_returns = pd.DataFrame({"daily_pnl": daily_pnl})
            df_returns["year"] = df_returns.index.year
            annual_return = df_returns.groupby("year")["daily_pnl"].sum()
        else:
            annual_return = pd.Series(dtype=float)

        # --- Compile result dataframe ---
        result = pd.DataFrame(
            {
                "spread": spread,
                "zscore": zscore,
                "position": position,
                "daily_pnl": daily_pnl,
                "cum_pnl": cum_pnl,
                "capital": capital_series,
                "total_return": total_return,
            }
        )

        # --- Plotting ---
        if plot and self.config.plot:
            fig, axs = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

            # 1. Underlyings
            axs[0].plot(self.y.index, self.y, label=self.tickers[0], color="red")
            axs[0].plot(self.x.index, self.x, label=self.tickers[1], color="blue")
            axs[0].set_title("Underlying Prices")
            axs[0].grid(True)
            axs[0].legend()

            # 2. Spread + signals + z-score corridor
            axs[1].plot(spread.index, spread, color="green", label="Spread")
            upper = roll_mean + entry_z * roll_std
            lower = roll_mean - entry_z * roll_std
            axs[1].fill_between(
                spread.index,
                lower,
                upper,
                color="grey",
                alpha=0.2,
                label=f"±{entry_z} Z-score",
            )
            buy_idx = result.index[result["position"] == 1]
            sell_idx = result.index[result["position"] == -1]
            axs[1].scatter(
                buy_idx,
                spread.loc[buy_idx],
                color="blue",
                marker="^",
                s=60,
                label="LONG",
            )
            axs[1].scatter(
                sell_idx,
                spread.loc[sell_idx],
                color="red",
                marker="v",
                s=60,
                label="SHORT",
            )
            axs[1].set_title("Spread with Signals and Z-score Corridor")
            axs[1].grid(True)
            axs[1].legend()

            # 3. Daily & cumulative PnL
            ax2 = axs[2].twinx()
            axs[2].plot(cum_pnl.index, cum_pnl, color="black", label="Cumulative PnL")
            ax2.plot(
                daily_pnl.index, daily_pnl, color="purple", alpha=0.7, label="Daily PnL"
            )
            axs[2].set_title("Cumulative PnL (Primary) & Daily PnL (Secondary)")
            axs[2].grid(True)
            axs[2].legend(loc="upper left")
            ax2.legend(loc="upper right")

            # 4. Annual returns as bar plot
            axs[3].bar(
                annual_return.index.astype(str),
                annual_return.values,
                color="skyblue",
            )
            axs[3].set_title("Annual Returns")
            axs[3].grid(True)

            plt.tight_layout()
            plt.show()

        return result
