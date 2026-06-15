# Backend — StratForge Backtest API

FastAPI service that swaps the frontend's synthetic simulator for **real OHLCV** data
from yfinance (default) or an uploaded CSV (NSE bhavcopy / generic OHLCV).

## Run

```bash
cd ~/Desktop/StratForge-Backtest/backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8765
```

API docs: http://localhost:8765/docs

## Endpoints

| Method | Path             | Body                          | Returns                         |
|--------|------------------|-------------------------------|---------------------------------|
| GET    | `/api/health`    | —                             | `{ok:true}`                     |
| POST   | `/api/backtest`  | frontend `config` dict        | results matching renderVals()   |
| POST   | `/api/upload`    | multipart `file=<csv>`        | `{upload_id, rows, symbols, …}` |

## Data sources

The frontend's existing **Data Source** segmented control drives backend behavior:

| UI choice    | `dataSource` value | Backend action                                                  |
|--------------|--------------------|------------------------------------------------------------------|
| Sample       | `sample`           | No backend call — JS simulator runs in-browser                  |
| API key      | `api`              | yfinance download (cached as Parquet under `cache/`)            |
| Upload CSV   | `csv`              | Reads previously uploaded CSV by `uploadId`                     |

NSE symbols (RELIANCE, TCS, INFY, …) are mapped to `.NS` for yfinance automatically.

## CSV formats accepted

1. **Generic OHLCV**: `date,open,high,low,close,volume[,symbol]`
2. **NSE bhavcopy** (cm*bhav.csv): `SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,…,TIMESTAMP`

Multi-symbol files are fine — the engine filters by the symbol it's loading.
