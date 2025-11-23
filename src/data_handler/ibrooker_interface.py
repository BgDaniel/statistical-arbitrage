from ib_insync import IB, Stock
import pandas as pd
import logging
from typing import Optional

# -----------------------------
# Constants
# -----------------------------
DEFAULT_EXCHANGE: str = "SMART"
DEFAULT_CURRENCY: str = "USD"
BAR_SIZE_DAILY: str = "1 day"
WHAT_TO_SHOW_CLOSE: str = "TRADES"
USE_RTH: bool = True
FORMAT_DATE: int = 1

logger = logging.getLogger(__name__)


class IBrokerConnector:
    """
    Interactive Brokers connector for fetching historical data (synchronous version).

    Attributes:
        host (str): Host IP for TWS/IB Gateway (default "127.0.0.1").
        port (int): Port for TWS/IB Gateway (default 7496 for live, 7497 for paper).
        client_id (int): Unique client ID for the connection.
        ib (IB): IB instance for communication.
    """

    def __init__(
        self, host: str = "127.0.0.1", port: int = 7496, client_id: int = 1
    ) -> None:
        """
        Initialize the IBrokerConnector.

        Args:
            host (str): Host IP for TWS/IB Gateway.
            port (int): Port for TWS/IB Gateway.
            client_id (int): Unique client ID.
        """
        self.ib: IB = IB()
        self.host: str = host
        self.port: int = port
        self.client_id: int = client_id

    def connect(self) -> None:
        """Connect to IB synchronously and log status."""
        self.ib.connect(self.host, self.port, clientId=self.client_id)
        if self.ib.isConnected():
            logger.info(
                f"Connected to IB at {self.host}:{self.port} with clientId={self.client_id}"
            )
        else:
            logger.error(
                f"Failed to connect to IB at {self.host}:{self.port} with clientId={self.client_id}"
            )

    def get_daily_close(
        self,
        ticker: str,
        end_date: pd.Timestamp,
        exchange: str = DEFAULT_EXCHANGE,
        currency: str = DEFAULT_CURRENCY,
    ) -> pd.Series:
        """
        Fetch historical daily closing prices for a given ticker.

        Args:
            ticker (str): Ticker symbol (e.g., "SPY").
            end_date (pd.Timestamp): End of historical data range.
            exchange (str): Exchange to pull data from (default "SMART").
            currency (str): Currency of the contract (default "USD").

        Returns:
            pd.Series: Daily closing prices indexed by datetime.
        """
        contract: Stock = Stock(ticker, exchange, currency)
        self.ib.qualifyContracts(contract)

        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime=end_date.strftime("%Y%m%d %H:%M:%S"),
            durationStr="1 Y",
            barSizeSetting=BAR_SIZE_DAILY,
            whatToShow=WHAT_TO_SHOW_CLOSE,
            useRTH=USE_RTH,
            formatDate=FORMAT_DATE,
        )

        if not bars:
            logger.warning(f"No historical data fetched for {ticker}")
            return pd.Series(dtype=float)

        df: pd.DataFrame = pd.DataFrame(
            [{"datetime": b.date, "close": b.close} for b in bars]
        )
        df.set_index("datetime", inplace=True)
        df = df[df.index <= end_date]
        return df["close"]

    def disconnect(self) -> None:
        """Disconnect from IB and log status."""
        self.ib.disconnect()
        logger.info("Disconnected from IB.")


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    connector = IBrokerConnector()
    connector.connect()

    end_ts: pd.Timestamp = pd.Timestamp("2025-11-21")
    close_series: pd.Series = connector.get_daily_close("SPY", end_ts)

    print(close_series.head())
    connector.disconnect()
