# data/binance_ingestor.py – Live Binance Stream Ingestor (WebSocket)

import asyncio
import json
import websockets
import logging
from datetime import datetime

class BinanceIngestor:
    def __init__(self):
        self.ws_url = (
            "wss://stream.binance.com:9443/stream?streams="
            "btcusdt@ticker/ethusdt@ticker/ethbtc@ticker"
        )
        self.latest = {}

    async def stream(self, callback):
        logging.info("🔌 Connecting to Binance WebSocket...")
        async with websockets.connect(self.ws_url) as websocket:
            async for msg in websocket:
                data = json.loads(msg)
                symbol = data["stream"].split("@")[0].upper()
                payload = data["data"]
                price = float(payload["c"])

                self.latest[symbol] = {
                    "price": price,
                    "timestamp": datetime.utcnow()
                }

                if all(k in self.latest for k in ["BTCUSDT", "ETHUSDT", "ETHBTC"]):
                    btc_price = self.latest["BTCUSDT"]["price"]
                    eth_price = self.latest["ETHUSDT"]["price"]
                    ethbtc_price = self.latest["ETHBTC"]["price"]
                    timestamp = self.latest["BTCUSDT"]["timestamp"]

                    callback(timestamp, btc_price, eth_price, ethbtc_price)

if __name__ == "__main__":
    ingestor = BinanceIngestor()
    async def print_callback(ts, btc, eth, ethbtc):
        print(f"[{ts}] BTC: {btc} ETH: {eth} ETHBTC: {ethbtc}")
    asyncio.run(ingestor.stream(print_callback))
