# Strategy specification — StratForge Backtest Lab

This is a portable spec you can paste to any AI (Claude, ChatGPT, Gemini, etc.)
so it produces a strategy that drops straight into the Lab editor with **zero edits**.

---

## Prompt template — copy from here

> You are writing a Python strategy for the **StratForge Backtest Lab**.
> The Lab will execute your code on real OHLCV data and produce trade-level results.
>
> ### Hard contract
> Write a **single top-level function** with this exact signature:
>
> ```python
> def signals(df, params):
>     ...
>     return enter_long, exit_long
> ```
>
> - `df` is a pandas DataFrame with columns `open, high, low, close, volume`,
>   indexed by datetime (tz-naive UTC). Intraday timestamps include hours/minutes.
> - `params` is a `dict` carrying the UI slider values (may be empty for Custom).
> - Return a **tuple of two boolean pandas Series**, both aligned to `df.index`.
>   `enter_long[i] == True` → open a long position at bar `i`.
>   `exit_long[i] == True`  → close any open long at bar `i`.
>
> ### What's already in scope (do not redefine)
> - `pd` (pandas), `np` (numpy)
> - `rsi(series, length)` → RSI as a Series
> - `sma(series, n)` → simple moving average
> - `ema(series, n)` → exponential moving average
> - `crossover(a, b)` → True at the bar where `a` crosses above `b`
> - `crossunder(a, b)` → True at the bar where `a` crosses below `b`
>
> `b` may be a Series **or** a scalar number.
>
> ### Imports allowed
> `import pandas as pd`, `import numpy as np`, `import math` — that's it.
> No file I/O, no network calls, no other modules.
>
> ### What you must NOT do
> - Do **not** hard-code symbols, dates, capital, position size, commission,
>   slippage, stop-loss, or take-profit. Those come from the UI sliders.
> - Do **not** define a `class Strategy` or call `run(...)` — only `signals()` is invoked.
> - Do **not** print or log; just return the two Series.
> - Do **not** call `df.ta.<anything>()` (no `pandas_ta`); use only the helpers above
>   or compute inline with rolling/shift.
>
> ### What you may do
> - Define helper functions above `signals()` — they share the same scope.
> - Use `df.rolling`, `df.shift`, `df.where`, `df.groupby(df.index.date)` for
>   intraday session logic, etc.
> - Use `params.get("key", default)` to read UI sliders safely.
>
> ### Strategy I want
> > **[ describe your strategy here in 2–5 sentences — entry condition, exit
> > condition, any filters (volume / volatility / trend), and whether it is
> > intended for daily bars or intraday ]**
>
> Output **only** the Python code, ready to paste into the editor.
> No prose around it, no markdown fences.

---

## Example output (what a good response looks like)

```python
# 10/30 SMA crossover, long-only, with a 200-SMA trend filter.
# Daily bars. Stop-loss / take-profit come from the UI risk panel.

def signals(df, params):
    c = df['close']
    fast = sma(c, 10)
    slow = sma(c, 30)
    trend = c > sma(c, 200)

    enter = crossover(fast, slow) & trend
    exit_ = crossunder(fast, slow)
    return enter, exit_
```

## Example — intraday ORB (NSE, 15m bars)

```python
# Opening Range Breakout — first 15 minutes of the NSE session.
# Long when 15m close breaks above the day's opening-range high.
# Exit at end of day or when close breaks below the opening-range low.

import pandas as pd, numpy as np

def signals(df, params):
    t = df.index.time
    d = df.index.date
    open_t = pd.Timestamp('09:15').time()
    end_t  = pd.Timestamp('09:30').time()
    sqoff  = pd.Timestamp('15:15').time()

    in_or = (t >= open_t) & (t < end_t)
    orh = df['high'].where(in_or).groupby(d).transform('max')
    orl = df['low' ].where(in_or).groupby(d).transform('min')
    orh = pd.Series(orh, index=df.index).groupby(d).ffill()
    orl = pd.Series(orl, index=df.index).groupby(d).ffill()

    after_or = t >= end_t
    enter = crossover(df['close'], orh) & after_or
    exit_ = (df['close'] < orl) | (t >= sqoff)
    return enter, exit_
```

## How to use the spec

1. Copy the prompt-template block above into any AI chat.
2. Replace the `[ describe your strategy here ]` placeholder with your idea.
3. Paste the AI's response into the Lab editor (Custom strategy → ✎ Edit code).
4. Set Universe, Timeframe, Period, Capital, SL/TP in the UI panels.
5. Hit Run Backtest.

If the response includes `class MyStrategy` or `run(...)` or `from backtest import`,
strip those — they came from older / generic Python backtest libraries and the Lab
ignores them. The only thing that matters is `def signals(df, params)`.
