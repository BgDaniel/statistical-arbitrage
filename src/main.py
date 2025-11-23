from data_handler.data_handler import DataHandler
from models.pair_arbitrage import PairArbitrage
from models.config import ArbitrageConfig


def main():
    tickers = ["QQQ", "PSQ"]

    dh = DataHandler(
        tickers=tickers,
        start_date="2020-01-01",
        end_date="2025-11-01"
    )
    prices = dh.fetch_data()

    # Prepare dict for PairArbitrage

    pair_data = {
        "QQQ": prices["QQQ"],
        "PSQ": prices["PSQ"],
    }

    config = ArbitrageConfig(window=120, plot=True)

    arb = PairArbitrage(pair_data, config)
    result = arb.analyze(plot=True)
    arb.backtest_strategy(entry_z=2.0, exit_z=1.0, plot=True)

    print("\nFinished.")
    print("Keys:", result.keys())


if __name__ == "__main__":
    main()
