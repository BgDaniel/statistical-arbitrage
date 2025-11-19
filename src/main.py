from data_handler.data_handler import DataHandler
from models.pair_arbitrage import PairArbitrage
from models.config import ArbitrageConfig


def main():
    tickers = ["AAPL", "MSFT"]

    dh = DataHandler(
        tickers=tickers,
        start_date="2020-01-01",
        end_date="2025-11-01"
    )
    prices = dh.fetch_data()

    # Prepare dict for PairArbitrage
    pair_data = {
        "AAPL": prices["AAPL"],
        "MSFT": prices["MSFT"],
    }

    config = ArbitrageConfig(window=120, plot=True)

    arb = PairArbitrage(pair_data, config)
    result = arb.analyze(plot=True)
    arb.backtest_strategy(plot=True)

    print("\nFinished.")
    print("Keys:", result.keys())


if __name__ == "__main__":
    main()
