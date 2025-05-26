# data/binance_ingestor.py – Resilient Binance Stream Ingestor with Auto-Reconnect & DNS Protection

import asyncio
import json
import websockets
import logging
import socket
from datetime import datetime

class BinanceIngestor:
    def __init__(self):
        self.ws_url = (
            "wss://stream.binance.com:9443/stream?streams="
            "btcusdt@ticker/ethusdt@ticker/ethbtc@ticker"
        )
        self.latest = {}

    async def stream(self, callback):
        retry_delay = 3  # seconds
        max_delay = 60

        while True:
            try:
                logging.info(f"🔌 [Binance] Connecting to WebSocket...")
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=10
                ) as websocket:
                    logging.info("✅ [Binance] Connected. Listening for tick data...")
                    retry_delay = 3  # reset after success

                    async for msg in websocket:
                        try:
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

                        except Exception as parse_err:
                            logging.warning(f"⚠️ Error parsing message: {parse_err} | Raw: {msg}")

            except websockets.exceptions.ConnectionClosedError as e:
                logging.error(f"🔌 [Binance] Connection closed: {e} — Retrying in {retry_delay}s")
            except socket.gaierror as e:
                logging.error(f"🌍 DNS resolution failed: {e} — Check network or try a VPN")
            except ConnectionRefusedError as e:
                logging.error(f"🚫 Connection refused: {e} — Binance may be blocking or offline")
            except Exception as e:
                logging.error(f"❌ [Binance] Unexpected error: {e}")

            logging.info(f"🔁 Retrying WebSocket connection in {retry_delay}s...")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_delay)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ingestor = BinanceIngestor()

    async def print_callback(ts, btc, eth, ethbtc):
        print(f"[{ts}] BTC: {btc:.2f} | ETH: {eth:.2f} | ETHBTC: {ethbtc:.6f}")

    asyncio.run(ingestor.stream(print_callback))
