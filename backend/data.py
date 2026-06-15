"""OHLCV loaders: yfinance (cached to parquet) + CSV upload (NSE bhavcopy / generic)."""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = CACHE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

log = logging.getLogger("data")

# NSE symbols need .NS suffix on Yahoo Finance.
NSE_SYMS = {
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "SBIN", "BHARTIARTL", "ITC",
}

# yfinance interval mapping
TF_MAP = {
    "1D": "1d",
    "1h": "60m",
    "30m": "30m",
    "15m": "15m",
    "5m": "5m",
}


def _yahoo_symbol(sym: str) -> str:
    return f"{sym}.NS" if sym.upper() in NSE_SYMS else sym


def _cache_path(sym: str, timeframe: str) -> Path:
    safe = sym.replace("/", "_")
    return CACHE_DIR / f"{safe}_{timeframe}.parquet"


def _within(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if df.empty:
        return df
    return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]


def load_yfinance(symbol: str, start: str, end: str, timeframe: str = "1D") -> pd.DataFrame:
    """Return OHLCV indexed by tz-naive UTC date. Cache hits are returned untouched."""
    import yfinance as yf  # imported lazily so server can boot without internet

    interval = TF_MAP.get(timeframe, "1d")
    cache = _cache_path(symbol, timeframe)
    if cache.exists():
        df = pd.read_parquet(cache)
        sub = _within(df, start, end)
        if not sub.empty:
            return sub

    ysym = _yahoo_symbol(symbol)
    log.info("yfinance download %s (%s) %s..%s", ysym, interval, start, end)
    raw = yf.download(
        ysym, start=start, end=end, interval=interval,
        auto_adjust=True, progress=False, threads=False,
    )
    if raw is None or raw.empty:
        raise ValueError(f"yfinance returned no data for {ysym}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.lower)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw = raw[["open", "high", "low", "close", "volume"]].dropna()

    raw.to_parquet(cache)
    return _within(raw, start, end)


def load_csv_upload(upload_id: str, symbol: Optional[str] = None) -> pd.DataFrame:
    """Read a previously uploaded CSV by upload_id."""
    path = UPLOAD_DIR / f"{upload_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"upload {upload_id} not found")
    df = pd.read_parquet(path)
    if symbol and "symbol" in df.columns:
        df = df[df["symbol"].str.upper() == symbol.upper()]
        df = df.drop(columns=["symbol"])
    return df


def save_csv_upload(file_bytes: bytes, filename: str) -> dict:
    """Parse generic OHLCV CSV or NSE bhavcopy and save as parquet keyed by upload_id."""
    import uuid

    df = _parse_csv_any(file_bytes)
    upload_id = uuid.uuid4().hex[:12]
    out = UPLOAD_DIR / f"{upload_id}.parquet"
    df.to_parquet(out)
    symbols = sorted(df["symbol"].unique().tolist()) if "symbol" in df.columns else []
    return {
        "upload_id": upload_id,
        "filename": filename,
        "rows": int(len(df)),
        "symbols": symbols,
        "date_min": str(df.index.min().date()) if len(df) else None,
        "date_max": str(df.index.max().date()) if len(df) else None,
    }


def _parse_csv_any(file_bytes: bytes) -> pd.DataFrame:
    """Auto-detect NSE bhavcopy vs generic OHLCV CSV. Returns df indexed by date."""
    text = file_bytes.decode("utf-8", errors="ignore")
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().upper() for c in df.columns]

    # NSE bhavcopy: SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,...
    if {"SYMBOL", "OPEN", "HIGH", "LOW", "CLOSE", "TIMESTAMP"}.issubset(df.columns):
        if "SERIES" in df.columns:
            df = df[df["SERIES"].astype(str).str.strip().isin(["EQ", "BE"])]
        vol_col = "TOTTRDQTY" if "TOTTRDQTY" in df.columns else "VOLUME"
        out = pd.DataFrame({
            "symbol": df["SYMBOL"].astype(str).str.upper().str.strip(),
            "open": df["OPEN"].astype(float),
            "high": df["HIGH"].astype(float),
            "low": df["LOW"].astype(float),
            "close": df["CLOSE"].astype(float),
            "volume": df[vol_col].astype(float) if vol_col in df.columns else 0.0,
        })
        out.index = pd.to_datetime(df["TIMESTAMP"], format="mixed", dayfirst=True, errors="coerce")
        out.index.name = "date"
        return out.dropna().sort_index()

    # generic OHLCV: date,open,high,low,close,volume[,symbol]
    rename = {"DATE": "date", "OPEN": "open", "HIGH": "high", "LOW": "low",
              "CLOSE": "close", "VOLUME": "volume", "SYMBOL": "symbol"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "date" not in df.columns:
        raise ValueError("CSV missing 'date' column")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    cols = [c for c in ["symbol", "open", "high", "low", "close", "volume"] if c in df.columns]
    return df[cols]


def load_for_symbol(symbol: str, start: str, end: str, timeframe: str,
                    source: str, upload_id: Optional[str] = None) -> pd.DataFrame:
    """Unified loader. source: 'yfinance' | 'csv'."""
    if source == "csv":
        if not upload_id:
            raise ValueError("csv source requires upload_id")
        df = load_csv_upload(upload_id, symbol=symbol)
    else:
        df = load_yfinance(symbol, start, end, timeframe)
    return _within(df, start, end)
