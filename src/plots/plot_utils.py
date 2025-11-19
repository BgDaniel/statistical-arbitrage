import matplotlib.pyplot as plt
import pandas as pd


def plot_beta_and_spread(
    beta: pd.Series,
    spread: pd.Series,
    title: str = "Beta and Spread"
) -> None:
    """
    Plot beta and spread in a single figure with two vertically stacked subplots.
    Beta = blue (top), Spread = red (bottom).

    Args:
        beta: Time-varying hedge ratio series.
        spread: Computed spread series.
        title: Title of the full figure.
    """

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(title, fontsize=14)

    # ---- Plot Beta (top) ----
    axes[0].plot(beta.index, beta.values, label="Beta", linewidth=1.5)
    axes[0].set_ylabel("Beta")
    axes[0].grid(True)
    axes[0].legend()

    # ---- Plot Spread (bottom) RED ----
    axes[1].plot(spread.index, spread.values, color="red", label="Spread", linewidth=1.5)
    axes[1].set_ylabel("Spread")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()
