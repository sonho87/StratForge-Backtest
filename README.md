# StratForge — Backtest Lab

Pixel-faithful implementation of the **Backtest Lab** design exported from Claude Design.
This is a UI-only prototype: workflow is end-to-end on a deterministic synthetic OHLCV
simulator. No live data fetching, no real Pine/Python execution — those come next, once
the design is locked.

## Run locally

```bash
# any static file server works; e.g.
python3 -m http.server 5500
# then open http://localhost:5500/
```

Opening `index.dc.html` directly via `file://` will not work — the DC runtime
boots off the URL path and needs an HTTP origin for the React/CDN fetch.

## What's in the box

| File              | Purpose                                                          |
|-------------------|------------------------------------------------------------------|
| `index.dc.html`   | The Backtest Lab design — markup, styles, and component logic    |
| `support.js`      | DC runtime (auto-loads React UMD, mounts the component)          |
| `index.html`      | Redirect to `index.dc.html` so `/` works                         |
| `DESIGN_README.md`| Original handoff README from Claude Design                        |

## Tabs

- **Strategy Lab** — code editor (Pine v5 / Python), universe & market, timeframe,
  data source (Sample / API key / CSV), period & capital, strategy params, risk.
- **Performance** — hero KPIs, equity curve vs B&H, drawdown, rolling Sharpe,
  go-live readiness gate, monthly-returns heatmap, win/loss donut, secondary metrics.
- **Trade Log** — per-trade stats, P&L bar strip, filterable trade table.
- **Markets** — capital allocation donut, market split, risk/return scatter,
  per-symbol performance table.

## Next steps (backend, after design sign-off)

The API key and CSV panels are scaffolding — they capture inputs and show status,
but the prototype runs on a deterministic seeded simulator. To go live:

1. Wire one real data source (Polygon / Alpaca / Zerodha Kite / Upstox / Yahoo,
   or a Yahoo/NSE CSV path as the cheapest first step).
2. Replace `runBacktest()` in `index.dc.html` with a real strategy execution
   engine — Pine via TradingView's broker bridge, or Python via a sandboxed
   `backtesting.py` / `vectorbt` worker.
