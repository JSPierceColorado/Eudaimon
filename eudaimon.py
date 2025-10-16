"""
Eudaimon — Momentum Rider (Full Alpaca Automation, no Telegram)

Philosophy:
  • Scan S&P 500 + leveraged bull/bear ETFs on 15m (or 1h) bars.
  • Enter strength (breakouts). Exit with TP/SL or timeout.
  • Each new position uses 10% of CURRENT buying power (by notional). No caps.

Env vars:
  ALPACA_API_KEY=...
  ALPACA_SECRET_KEY=...
  ALPACA_PAPER=true|false

  # Leveraged products list (comma-separated)
  LEVERAGED_TICKERS="TQQQ,SQQQ,SPXL,SPXS,SOXL,SOXS,FNGU,FNGD,LABU,LABD,RETL,RETD,NUGT,DUST,BOIL,KOLD,UVXY,VIXY,SVIX"

  # Strategy knobs
  TIMEFRAME_MIN=15                  # set 60 for hourly
  BREAKOUT_LOOKBACK=20              # bars
  VOL_RUNRATE_MULT=1.5              # min run-rate vs 20-bar avg
  RSI_MIN=55                        # momentum zone lower bound

  # Exits
  TP_PCT=6                          # +6% take-profit
  SL_PCT=3                          # -3% stop-loss
  TIMEOUT_DAYS=3                    # flat/sideways timeout

  # Sizing
  ALLOW_FRACTIONAL=true|false
  NOTIONAL_PCT_PER_TRADE=10

  # Cadence
  SCAN_SECONDS=900
  MANAGE_SECONDS=60

  # State
  STATE_PATH=/mnt/data/eudaimon_state.json
"""
from __future__ import annotations
import os, json, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import pandas_ta as ta

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, Adjustment

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() in {"1","true","yes"}

LEVERAGED_TICKERS = [s.strip().upper() for s in os.getenv(
    "LEVERAGED_TICKERS",
    "TQQQ,SQQQ,SPXL,SPXS,SOXL,SOXS,FNGU,FNGD,LABU,LABD,RETL,RETD,NUGT,DUST,BOIL,KOLD,UVXY,VIXY,SVIX"
).split(",") if s.strip()]

TIMEFRAME_MIN = int(os.getenv("TIMEFRAME_MIN", "15"))
BREAKOUT_LOOKBACK = int(os.getenv("BREAKOUT_LOOKBACK", "20"))
VOL_RUNRATE_MULT = float(os.getenv("VOL_RUNRATE_MULT", "1.5"))
RSI_MIN = int(os.getenv("RSI_MIN", "55"))

TP_PCT = float(os.getenv("TP_PCT", "6"))
SL_PCT = float(os.getenv("SL_PCT", "3"))
TIMEOUT_DAYS = int(os.getenv("TIMEOUT_DAYS", "3"))

ALLOW_FRACTIONAL = os.getenv("ALLOW_FRACTIONAL", "true").lower() in {"1","true","yes"}
NOTIONAL_PCT_PER_TRADE = float(os.getenv("NOTIONAL_PCT_PER_TRADE", "10"))

SCAN_SECONDS = int(os.getenv("SCAN_SECONDS", "900"))
MANAGE_SECONDS = int(os.getenv("MANAGE_SECONDS", "60"))

STATE_PATH = Path(os.getenv("STATE_PATH", "/mnt/data/eudaimon_state.json"))

TIMEFRAME = TimeFrame(TIMEFRAME_MIN, TimeFrameUnit.Minute)
LOOKBACK_DAYS_STOCK = 30

BOT_TAG = "EUDAIMON"

# ──────────────────────────────────────────────────────────────────────────────
# Clients
# ──────────────────────────────────────────────────────────────────────────────
if not (ALPACA_API_KEY and ALPACA_SECRET_KEY):
    raise SystemExit("Missing ALPACA credentials")

stock_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
trading = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)

print("[INFO] Using data feed: IEX")

# ──────────────────────────────────────────────────────────────────────────────
# Universe (GitHub source → fallback)
# ──────────────────────────────────────────────────────────────────────────────
FALLBACK_SP500 = ["AAPL","MSFT","AMZN","GOOGL","META","NVDA","JPM","AVGO","SPY"]

GITHUB_SP500_URL = os.getenv(
    "GITHUB_SP500_URL",
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
)

def load_sp500_from_github() -> List[str]:
    df = pd.read_csv(GITHUB_SP500_URL)
    col = next(c for c in df.columns if str(c).strip().lower() in {"symbol", "ticker"})
    syms = (
        df[col]
        .dropna()
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(".", "-", regex=False)
        .unique()
        .tolist()
    )
    cleaned = [s for s in syms if len(s) >= 1 and all(ch.isalnum() or ch == "-" for ch in s)]
    return sorted(cleaned)

def load_sp500() -> List[str]:
    try:
        return load_sp500_from_github()
    except Exception as e:
        print(f"[WARN] load_sp500 GitHub failed: {e}. Falling back to static list.")
        return FALLBACK_SP500

def full_universe() -> List[str]:
    return sorted(set(load_sp500()) | set(LEVERAGED_TICKERS))

# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────
def load_state() -> Dict[str,Any]:
    try:
        if STATE_PATH.exists():
            return json.loads(STATE_PATH.read_text())
    except Exception:
        pass
    return {"positions": {}}

def save_state(state: Dict[str,Any]):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state))

state = load_state()

# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────
def to_df_from_response(resp, symbol: str) -> pd.DataFrame:
    if hasattr(resp, "data") and resp.data:
        bars_list = resp.data.get(symbol, [])
        rows = [{
            "t": b.timestamp,
            "o": float(b.open),
            "h": float(b.high),
            "l": float(b.low),
            "c": float(b.close),
            "v": float(getattr(b, "volume", np.nan)) if getattr(b, "volume", None) is not None else np.nan,
        } for b in bars_list]
        return pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    if hasattr(resp, "df") and isinstance(resp.df, pd.DataFrame) and not resp.df.empty:
        df = resp.df.copy()
        if "symbol" in df.index.names:
            try:
                df = df.xs(symbol, level("symbol"))
            except Exception:
                pass
        df = df.reset_index().rename(columns={
            "timestamp": "t", "open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"
        })
        keep = ["t","o","h","l","c","v"]
        return df[keep].sort_values("t").reset_index(drop=True)
    return pd.DataFrame(columns=["t","o","h","l","c","v"])

def fetch_bars(symbol: str, end: datetime) -> pd.DataFrame:
    start = end - timedelta(days=LOOKBACK_DAYS_STOCK)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TIMEFRAME,
        start=start,
        end=end,
        adjustment=Adjustment.SPLIT,
        feed=DataFeed.IEX,
        limit=10000,
    )
    resp = stock_client.get_stock_bars(req)
    return to_df_from_response(resp, symbol)

# ──────────────────────────────────────────────────────────────────────────────
# Signals
# ──────────────────────────────────────────────────────────────────────────────
def runrate_volume_ok(v: pd.Series) -> bool:
    if len(v) < 21:
        return False
    avg20 = v.tail(20).mean()
    return v.iloc[-1] >= VOL_RUNRATE_MULT * avg20

def breakout_signal(df: pd.DataFrame) -> bool:
    if len(df) < max(BREAKOUT_LOOKBACK, 50):
        return False
    c = df["c"].astype(float)
    h = df["h"].astype(float)
    v = df["v"].astype(float)
    rsi14 = ta.rsi(c, length=14).iloc[-1]
    hh = h.tail(BREAKOUT_LOOKBACK).max()
    return (c.iloc[-1] > hh) and runrate_volume_ok(v) and (rsi14 >= RSI_MIN)

# ──────────────────────────────────────────────────────────────────────────────
# Orders
# ──────────────────────────────────────────────────────────────────────────────
def get_buying_power() -> float:
    acct = trading.get_account()
    return float(acct.buying_power)

def place_bracket_buy(symbol: str, last_price: float) -> str:
    notional = get_buying_power() * (NOTIONAL_PCT_PER_TRADE / 100.0)
    if notional <= 0:
        raise RuntimeError("No buying power available")

    tp_price = round(last_price * (1 + TP_PCT/100.0), 2)
    sl_price = round(last_price * (1 - SL_PCT/100.0), 2)
    client_oid = f"{BOT_TAG}-{symbol}-{int(datetime.now(timezone.utc).timestamp())}"

    if ALLOW_FRACTIONAL:
        order = trading.submit_order(
            order_data=MarketOrderRequest(
                symbol=symbol,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                notional=notional,
                client_order_id=client_oid,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=tp_price),
                stop_loss=StopLossRequest(stop_price=sl_price),
            )
        )
    else:
        qty = max(1, int(notional // max(last_price, 0.01)))
        order = trading.submit_order(
            order_data=MarketOrderRequest(
                symbol=symbol,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                qty=qty,
                client_order_id=client_oid,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=tp_price),
                stop_loss=StopLossRequest(stop_price=sl_price),
            )
        )
    return order.id

# ──────────────────────────────────────────────────────────────────────────────
# Timeout / management
# ──────────────────────────────────────────────────────────────────────────────
def is_timeout(opened_iso: str, entry_px: float, df: pd.DataFrame) -> bool:
    opened_at = datetime.fromisoformat(opened_iso)
    if datetime.now(timezone.utc) - opened_at < timedelta(days=TIMEOUT_DAYS):
        return False
    if df.empty:
        return False
    c = df["c"].astype(float)
    atr = ta.atr(df["h"], df["l"], df["c"], length=14).iloc[-1]
    net_move = abs(c.iloc[-1] - entry_px)
    return net_move < 1.5 * float(atr)

def close_position(symbol: str):
    try:
        trading.close_position(symbol)
        print(f"[INFO] Closed {symbol} on timeout")
    except Exception as e:
        print(f"[WARN] close_position failed for {symbol}: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Main loops
# ──────────────────────────────────────────────────────────────────────────────
def scan_and_trade():
    syms = full_universe()
    now = datetime.now(timezone.utc)
    print(f"[INFO] Eudaimon scan at {now.isoformat()} — {len(syms)} symbols")
    for sym in syms:
        try:
            df = fetch_bars(sym, now)
            if df.empty:
                continue
            if breakout_signal(df):
                last = float(df["c"].iloc[-1])
                oid = place_bracket_buy(sym, last)
                state["positions"][sym] = {
                    "opened_at": now.isoformat(),
                    "entry_px": last,
                    "order_id": oid,
                }
                print(f"[ALERT] EUDAIMON BUY {sym} @ ~{last:.2f} (oid={oid})")
                save_state(state)
        except Exception as e:
            print(f"[ERROR] {sym} scan/trade failed: {e}")

def manage_positions():
    try:
        positions = {p.symbol: p for p in trading.get_all_positions()}
        for sym, meta in list(state["positions"].items()):
            if sym not in positions:
                print(f"[INFO] Position closed: {sym}")
                del state["positions"][sym]
                continue
            df = fetch_bars(sym, datetime.now(timezone.utc))
            if not df.empty and is_timeout(meta["opened_at"], float(meta["entry_px"]), df):
                close_position(sym)
                del state["positions"][sym]
        save_state(state)
    except Exception as e:
        print(f"[WARN] manage_positions error: {e}")

def loop_forever():
    next_scan = 0
    while True:
        now_ts = time.time()
        if now_ts >= next_scan:
            scan_and_trade()
            next_scan = now_ts + SCAN_SECONDS
        manage_positions()
        time.sleep(MANAGE_SECONDS)

if __name__ == "__main__":
    loop_forever()
