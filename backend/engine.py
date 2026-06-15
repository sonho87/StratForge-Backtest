"""Backtest engine — real signals on real OHLCV.

Returns a JSON-serializable dict whose shape matches what renderVals() in
index.dc.html consumes: days, eq, bench, dd, roll, trades, monthly, perSym, m.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data import load_for_symbol

# ---------- indicators ----------

def rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(length).mean()
    down = (-delta.clip(upper=0)).rolling(length).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def crossover(a: pd.Series, b: pd.Series | float) -> pd.Series:
    if isinstance(b, (int, float)):
        return (a.shift(1) < b) & (a >= b)
    return (a.shift(1) < b.shift(1)) & (a >= b)


def crossunder(a: pd.Series, b: pd.Series | float) -> pd.Series:
    if isinstance(b, (int, float)):
        return (a.shift(1) > b) & (a <= b)
    return (a.shift(1) > b.shift(1)) & (a <= b)


# ---------- signal generation ----------

def run_custom_python(code: str, df: pd.DataFrame, params: dict) -> tuple[pd.Series, pd.Series]:
    """Execute user-pasted Python. The pasted code must define `signals(df, params)`
    returning `(enter_long, exit_long)` boolean Series aligned to df.index.

    Sandbox: no builtins beyond a safe whitelist; no imports needed (pd/np/ta exposed).
    Runs in-process — fine for local single-user use; wrap in a subprocess for prod.
    """
    import math as _math
    _ALLOWED_IMPORTS = {"pandas": pd, "numpy": np, "math": _math, "pd": pd, "np": np}

    def _safe_import(name, *a, **kw):
        root = name.split(".")[0]
        if root in _ALLOWED_IMPORTS:
            return _ALLOWED_IMPORTS[root]
        raise ImportError(f"import of '{name}' is not allowed in custom strategies "
                          f"(allowed: pandas, numpy, math)")

    safe_builtins = {
        "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
        "range": range, "round": round, "int": int, "float": float, "bool": bool,
        "list": list, "dict": dict, "tuple": tuple, "set": set, "str": str,
        "enumerate": enumerate, "zip": zip, "any": any, "all": all,
        "True": True, "False": False, "None": None,
        "__import__": _safe_import, "__build_class__": __build_class__,
        "print": print, "isinstance": isinstance, "type": type, "getattr": getattr,
        "hasattr": hasattr, "setattr": setattr,
    }
    env = {
        "__builtins__": safe_builtins,
        "pd": pd, "np": np,
        "rsi": rsi, "sma": sma, "ema": ema,
        "crossover": crossover, "crossunder": crossunder,
    }
    exec(compile(code, "<custom_strategy>", "exec"), env, env)
    fn = env.get("signals")
    if not callable(fn):
        raise ValueError("Custom Python must define a function: signals(df, params) -> (enter, exit)")
    out = fn(df, params)
    if not (isinstance(out, tuple) and len(out) == 2):
        raise ValueError("signals(df, params) must return a tuple (enter_long, exit_long)")
    enter, exit_ = out
    return pd.Series(enter, index=df.index).fillna(False), pd.Series(exit_, index=df.index).fillna(False)


def signals(df: pd.DataFrame, strategy: str, params: dict,
            custom_code: str = "", custom_lang: str = "") -> tuple[pd.Series, pd.Series]:
    """Return (enter_long, exit_long) boolean series aligned to df.index."""
    if strategy == "custom" and custom_code.strip() and custom_lang == "python":
        return run_custom_python(custom_code, df, params)
    c = df["close"]
    if strategy == "rsiMeanRev":
        r = rsi(c, int(params.get("rsiLen", 14)))
        return crossover(r, float(params.get("oversold", 30))), \
               crossunder(r, float(params.get("overbought", 70)))
    if strategy == "maCross":
        f, s = sma(c, int(params["fastMA"])), sma(c, int(params["slowMA"]))
        return crossover(f, s), crossunder(f, s)
    if strategy == "macd":
        fast = ema(c, int(params["fast"]))
        slow = ema(c, int(params["slow"]))
        macd_line = fast - slow
        sig = ema(macd_line, int(params["signal"]))
        return crossover(macd_line, sig), crossunder(macd_line, sig)
    if strategy == "bollinger":
        n = int(params["bbLen"])
        mult = float(params["bbStd"])
        basis = sma(c, n)
        dev = mult * c.rolling(n).std()
        return crossover(c, basis + dev), crossunder(c, basis)
    # custom — z-score breakout
    lb = int(params.get("lookback", 20))
    thr = float(params.get("threshold", 1.5))
    z = (c - sma(c, lb)) / c.rolling(lb).std()
    return crossover(z, thr), crossunder(z, 0)


# ---------- per-symbol simulation ----------

def simulate_symbol(df: pd.DataFrame, cfg: dict) -> dict:
    """Walk bars; enter on signal, exit on opposite signal / SL / TP. Returns trades + per-bar equity."""
    enter, exit_ = signals(df, cfg["strategy"], cfg["params"],
                            custom_code=cfg.get("customCode", ""),
                            custom_lang=cfg.get("customLang", ""))
    closes = df["close"].values
    dates = df.index.to_pydatetime()
    n = len(closes)

    stop = float(cfg["stopLoss"]) / 100.0
    target = float(cfg["takeProfit"]) / 100.0
    pos_frac = max(0.1, min(1.0, float(cfg["posSize"]) / 100.0))
    rt_cost = (float(cfg["commission"]) + float(cfg["slippage"])) / 100.0 * 2

    trades: List[dict] = []
    equity = 1.0
    eq = np.empty(n)
    in_pos = False
    entry_px = 0.0
    entry_i = 0

    enter_arr = enter.fillna(False).values
    exit_arr = exit_.fillna(False).values

    for i in range(n):
        px = closes[i]
        if in_pos:
            ret = (px - entry_px) / entry_px
            unreal = ret * pos_frac
            eq[i] = equity * (1 + unreal)
            hit_stop = ret <= -stop
            hit_target = ret >= target
            if hit_stop or hit_target or exit_arr[i] or i == n - 1:
                realised = (-stop if hit_stop else (target if hit_target else ret))
                net = realised * pos_frac - rt_cost * pos_frac
                before = equity
                equity = equity * (1 + net)
                trades.append({
                    "entry_i": entry_i, "exit_i": i,
                    "entry_date": dates[entry_i], "exit_date": dates[i],
                    "bars": i - entry_i, "ret": float(net),
                    "pnl": float(equity - before), "equity_after": float(equity),
                })
                eq[i] = equity
                in_pos = False
        else:
            eq[i] = equity
            if enter_arr[i]:
                in_pos = True
                entry_px = px
                entry_i = i
    return {"dates": dates, "equity": eq, "trades": trades, "close": closes}


# ---------- portfolio aggregation ----------

def run_backtest(cfg: dict) -> dict:
    symbols = cfg.get("symbols") or ["AAPL"]
    start = f"{cfg['startYear']}-01-01"
    end = f"{cfg['endYear']}-12-31"
    timeframe = cfg.get("timeframe", "1D")
    source = cfg.get("_source", "yfinance")
    upload_id = cfg.get("_upload_id")

    per_sym_data = {}
    for sym in symbols:
        try:
            df = load_for_symbol(sym, start, end, timeframe, source, upload_id)
            if len(df) < 30:
                continue
            per_sym_data[sym] = df
        except Exception as e:
            print(f"[warn] skipped {sym}: {e}")
            continue

    if not per_sym_data:
        raise ValueError("No usable price data — check symbols / source / date range.")

    # union timeline
    all_dates = sorted({d for df in per_sym_data.values() for d in df.index})
    timeline = pd.DatetimeIndex(all_dates)
    n = len(timeline)

    sims = {sym: simulate_symbol(df.reindex(timeline).ffill(), cfg) for sym, df in per_sym_data.items()}

    # equal-weight portfolio: avg per-symbol equity scaled to initial capital
    capital = float(cfg["capital"])
    eq_matrix = np.column_stack([s["equity"] for s in sims.values()])
    port_eq = capital * eq_matrix.mean(axis=1)

    # benchmark: equal-weight buy & hold (real prices)
    bench_matrix = np.column_stack([s["close"] / s["close"][0] for s in sims.values()])
    bench = capital * bench_matrix.mean(axis=1)

    # drawdown
    peak = np.maximum.accumulate(port_eq)
    dd = port_eq / peak - 1.0
    max_dd = float(dd.min())

    # underwater duration
    ddur = 0; cur = 0; max_dur = 0
    for v in dd:
        if v < 0:
            cur += 1; max_dur = max(max_dur, cur)
        else:
            cur = 0
    ddur = max_dur

    # returns / sharpe / sortino / vol
    rets = np.diff(port_eq) / port_eq[:-1]
    mu = float(np.mean(rets)) if len(rets) else 0.0
    sd = float(np.std(rets)) if len(rets) else 0.0
    downs = rets[rets < 0]
    dsd = float(np.sqrt(np.mean(downs**2))) if len(downs) else sd
    sharpe = (mu / sd) * math.sqrt(252) if sd else 0.0
    sortino = (mu / dsd) * math.sqrt(252) if dsd else 0.0
    vol = sd * math.sqrt(252)

    years = max(1e-9, n / 252.0)
    total_ret = port_eq[-1] / capital - 1
    cagr = (port_eq[-1] / capital) ** (1 / years) - 1
    bench_ret = bench[-1] / capital - 1
    bench_cagr = (bench[-1] / capital) ** (1 / years) - 1

    # rolling sharpe (126d)
    W = 126
    roll = [None] * n
    for i in range(W, n):
        w = rets[i - W:i]
        if len(w) and np.std(w):
            roll[i] = float(np.mean(w) / np.std(w) * math.sqrt(252))
        else:
            roll[i] = 0.0
    roll_vals = [r for r in roll if r is not None]
    avg_roll = float(np.mean(roll_vals)) if roll_vals else 0.0

    # aggregate trades across symbols, chronological
    all_trades = []
    eq_acc = capital
    for sym, sim in sims.items():
        for t in sim["trades"]:
            all_trades.append({"sym": sym, **t})
    all_trades.sort(key=lambda t: t["exit_date"])
    # recompute portfolio-level equity per trade (sequential allocation)
    for t in all_trades:
        before = eq_acc
        eq_acc = eq_acc * (1 + t["ret"] / max(1, len(sims)))  # split risk across syms
        t["equity"] = eq_acc
        t["pnl"] = eq_acc - before

    wins = [t for t in all_trades if t["ret"] > 0]
    losses = [t for t in all_trades if t["ret"] <= 0]
    sum_win = sum(t["pnl"] for t in wins)
    sum_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = (sum_win / sum_loss) if sum_loss else (9.99 if sum_win > 0 else 0.0)
    win_rate = (len(wins) / len(all_trades)) if all_trades else 0.0
    avg_win = float(np.mean([t["ret"] for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t["ret"] for t in losses])) if losses else 0.0
    avg_trade = float(np.mean([t["ret"] for t in all_trades])) if all_trades else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    best = max((t["ret"] for t in all_trades), default=0.0)
    worst = min((t["ret"] for t in all_trades), default=0.0)
    avg_bars = float(np.mean([t["bars"] for t in all_trades])) if all_trades else 0.0
    exposure = sum(t["bars"] for t in all_trades) / n if n else 0.0

    # monthly returns
    monthly: Dict[str, Dict[int, float]] = {}
    eq_series = pd.Series(port_eq, index=timeline)
    monthly_eq = eq_series.resample("ME").last()
    prev = capital
    for ts, val in monthly_eq.items():
        yr = str(ts.year); mo = ts.month - 1  # JS uses 0-indexed months
        ret = val / prev - 1
        monthly.setdefault(yr, {})[mo] = float(ret)
        prev = val

    # per-symbol stats
    per_sym = []
    for sym, sim in sims.items():
        eq_s = sim["equity"]
        ts = sim["trades"]
        years_s = max(1e-9, len(eq_s) / 252.0)
        cagr_s = (eq_s[-1] / eq_s[0]) ** (1 / years_s) - 1 if eq_s[0] else 0.0
        rets_s = np.diff(eq_s) / eq_s[:-1] if len(eq_s) > 1 else np.array([0.0])
        sd_s = float(np.std(rets_s)) if len(rets_s) else 0.0
        vol_s = sd_s * math.sqrt(252)
        sharpe_s = (float(np.mean(rets_s)) / sd_s) * math.sqrt(252) if sd_s else 0.0
        peak_s = np.maximum.accumulate(eq_s)
        maxdd_s = float((eq_s / peak_s - 1).min())
        wins_s = sum(1 for t in ts if t["ret"] > 0)
        win_rate_s = (wins_s / len(ts)) if ts else 0.0
        market = "NSE" if sym.upper() in {"RELIANCE", "TCS", "INFY", "HDFCBANK",
                                           "ICICIBANK", "SBIN", "BHARTIARTL", "ITC"} else "US"
        per_sym.append({
            "sym": sym, "market": market,
            "cagr": float(cagr_s), "sharpe": float(sharpe_s),
            "maxDD": float(-maxdd_s), "winRate": float(win_rate_s),
            "trades": len(ts), "vol": float(vol_s),
            "totalRet": float(eq_s[-1] / eq_s[0] - 1) if eq_s[0] else 0.0,
        })
    tot_contrib = sum(max(0, p["totalRet"]) for p in per_sym) or 1.0
    for p in per_sym:
        p["contribution"] = max(0, p["totalRet"]) / tot_contrib

    # trades serialisation
    trades_out = []
    for t in all_trades:
        trades_out.append({
            "sym": t["sym"], "side": "LONG",
            "entryIdx": int(np.searchsorted(timeline.values, np.datetime64(t["entry_date"]))),
            "exitIdx": int(np.searchsorted(timeline.values, np.datetime64(t["exit_date"]))),
            "bars": int(t["bars"]),
            "ret": float(t["ret"]), "pnl": float(t["pnl"]), "equity": float(t["equity"]),
        })

    return {
        "days": [d.isoformat() for d in timeline.to_pydatetime()],
        "eq": [float(x) for x in port_eq],
        "bench": [float(x) for x in bench],
        "dd": [float(x) for x in dd],
        "roll": roll,
        "trades": trades_out,
        "monthly": monthly,
        "perSym": per_sym,
        "m": {
            "totalRet": float(total_ret), "cagr": float(cagr),
            "sharpe": float(sharpe), "sortino": float(sortino), "vol": float(vol),
            "maxDD": float(max_dd), "maxDDdur": int(ddur),
            "exposure": float(exposure),
            "benchRet": float(bench_ret), "benchCagr": float(bench_cagr),
            "profitFactor": float(profit_factor),
            "winRate": float(win_rate),
            "avgWin": float(avg_win), "avgLoss": float(avg_loss),
            "avgTrade": float(avg_trade), "expectancy": float(expectancy),
            "bestTrade": float(best), "worstTrade": float(worst),
            "avgBars": float(avg_bars), "avgRoll": float(avg_roll),
            "nTrades": int(len(all_trades)),
            "years": float(years),
            "finalEq": float(port_eq[-1]), "finalBench": float(bench[-1]),
            "wins": int(len(wins)), "losses": int(len(losses)),
        },
    }
