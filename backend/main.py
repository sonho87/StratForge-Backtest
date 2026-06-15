"""FastAPI entrypoint. Run with:  uvicorn main:app --reload --port 8765"""
from __future__ import annotations

import logging
import traceback

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from data import save_csv_upload
from engine import run_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("api")

app = FastAPI(title="StratForge Backtest API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/backtest")
async def backtest(cfg: dict):
    """Run a backtest against real data. Body is the frontend's `config` dict."""
    try:
        ds = cfg.get("dataSource", "api")
        if ds == "csv":
            cfg["_source"] = "csv"
            cfg["_upload_id"] = cfg.get("uploadId")
        else:
            cfg["_source"] = "yfinance"
        return run_backtest(cfg)
    except Exception as e:
        log.exception("backtest failed")
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Accept NSE bhavcopy or generic OHLCV CSV. Returns upload_id used by /api/backtest."""
    try:
        content = await file.read()
        return save_csv_upload(content, file.filename or "upload.csv")
    except Exception as e:
        log.exception("upload failed")
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")
