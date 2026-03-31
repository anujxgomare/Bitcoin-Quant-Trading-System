"""
Bitcoin Quant Trading — FastAPI Backend
Run: uvicorn server:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio, os, time, threading, traceback
from datetime import datetime
from pathlib import Path

# ── import strategy module ────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

try:
    from bitcoin_quant_strategy import (
        CONFIG, fetch_ohlcv, compute_indicators, label_data,
        build_feature_matrix, train_model, backtest,
        predict_latest, plot_strategy_dashboard, plot_feature_importance,
    )
    import joblib, numpy as np
    STRATEGY_LOADED = True
except Exception as e:
    STRATEGY_LOADED = False
    STRATEGY_ERROR  = str(e)

app = FastAPI(title="BTC Quant API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── serve static files from current dir ──────────────────────────────────────
BASE = Path(__file__).parent
app.mount("/charts", StaticFiles(directory=str(BASE)), name="charts")

# ── in-memory state ───────────────────────────────────────────────────────────
STATE = {
    "status"        : "idle",      # idle | training | ready | error
    "last_update"   : None,
    "prediction"    : None,
    "backtest_rule" : None,
    "backtest_ml"   : None,
    "error"         : None,
    "trade"         : None,
    "trade_log"     : [],
}
_df = _model = _scaler = _xgb = _X = None


# =============================================================================
# HELPERS
# =============================================================================

def _position_size(capital: float, risk_pct: float,
                   entry: float, sl: float) -> dict:
    """Kelly-capped fixed-fractional position sizing."""
    risk_amount = capital * risk_pct
    risk_per_unit = abs(entry - sl)
    units = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
    notional = units * entry
    return {
        "units"       : round(units, 6),
        "notional_usd": round(notional, 2),
        "risk_usd"    : round(risk_amount, 2),
    }


def _breakeven(entry: float, fee_pct: float = 0.001) -> float:
    """Breakeven price after 2-leg fees (entry + exit)."""
    return round(entry * (1 + 2 * fee_pct), 2)


def _build_trade(pred: dict, capital: float = 10_000) -> dict | None:
    """Construct a demo trade object from a prediction."""
    direction = pred["prediction"]
    if direction == "FLAT":
        return None

    entry = pred["close"]
    sl    = pred["stop_loss"]
    tp    = pred["take_profit"]
    atr   = pred["atr"]

    if sl is None or tp is None:
        return None

    sizing = _position_size(capital, CONFIG["risk_pct"], entry, sl)
    risk   = abs(entry - sl)
    reward = abs(tp - entry)
    rr     = round(reward / risk, 2) if risk > 0 else 0
    be     = _breakeven(entry)

    return {
        "id"           : int(time.time()),
        "timestamp"    : pred["timestamp"],
        "direction"    : direction,
        "entry"        : entry,
        "stop_loss"    : sl,
        "take_profit"  : tp,
        "breakeven"    : be,
        "risk_reward"  : rr,
        "atr"          : atr,
        "risk_pct"     : CONFIG["risk_pct"] * 100,
        "risk_usd"     : sizing["risk_usd"],
        "units"        : sizing["units"],
        "notional_usd" : sizing["notional_usd"],
        "confidence"   : pred["confidence"],
        "composite"    : pred["composite_score"],
        "rsi"          : pred["rsi"],
        "macd_hist"    : pred["macd_hist"],
        "bb_squeeze"   : pred["bb_squeeze"],
        "vol_spike"    : pred["vol_spike"],
        "status"       : "open",
        "pnl_usd"      : None,
        "pnl_pct"      : None,
        "close_price"  : None,
        "close_reason" : None,
    }


def _run_full_pipeline():
    """Blocking pipeline — run in a background thread."""
    global _df, _model, _scaler, _xgb, _X

    STATE["status"] = "training"
    STATE["error"]  = None

    try:
        _df           = fetch_ohlcv(CONFIG)
        _df           = compute_indicators(_df, CONFIG)
        _df           = label_data(_df, CONFIG)
        _X, y         = build_feature_matrix(_df)
        _model, _scaler, _xgb = train_model(_X, y, CONFIG)

        trades_rule   = backtest(_df, CONFIG, use_ml=False)
        trades_ml     = backtest(_df, CONFIG, use_ml=True,
                                 model=_model, scaler=_scaler, X=_X)

        pred          = predict_latest(_df, _model, _scaler, CONFIG)
        trade         = _build_trade(pred)

        # Save charts
        plot_strategy_dashboard(_df, trades_rule, CONFIG)
        plot_feature_importance(_xgb, _X.columns.tolist())

        # Serialise DataFrames
        def _df_to_dict(df):
            if df is None or df.empty:
                return {"trades": [], "metrics": {}}
            wr = float(df["win"].mean())
            tr = float(df["pnl_pct"].sum())
            return {
                "metrics": {
                    "total_trades": len(df),
                    "win_rate"    : round(wr * 100, 2),
                    "total_return": round(tr * 100, 2),
                    "avg_pnl"     : round(float(df["pnl_pct"].mean()) * 100, 4),
                    "best_trade"  : round(float(df["pnl_pct"].max()) * 100, 4),
                    "worst_trade" : round(float(df["pnl_pct"].min()) * 100, 4),
                    "tp_count"    : int((df["result"] == "TP").sum()),
                    "sl_count"    : int((df["result"] == "SL").sum()),
                },
                "equity": [round(10000 * (1 + df["pnl_pct"].iloc[:i+1].sum()), 2)
                           for i in range(len(df))],
            }

        STATE["prediction"]    = pred
        STATE["backtest_rule"] = _df_to_dict(trades_rule)
        STATE["backtest_ml"]   = _df_to_dict(trades_ml)
        STATE["trade"]         = trade
        STATE["last_update"]   = datetime.utcnow().isoformat()
        STATE["status"]        = "ready"

        if trade:
            STATE["trade_log"].insert(0, {**trade, "log_time": datetime.utcnow().isoformat()})
            STATE["trade_log"] = STATE["trade_log"][:50]

    except Exception:
        STATE["status"] = "error"
        STATE["error"]  = traceback.format_exc()


# =============================================================================
# ROUTES
# =============================================================================

@app.get("/")
def root():
    return FileResponse(str(BASE / "index.html"))


@app.post("/api/run")
def trigger_run(background_tasks: BackgroundTasks):
    if STATE["status"] == "training":
        return {"message": "Already running"}
    background_tasks.add_task(_run_full_pipeline)
    return {"message": "Pipeline started"}


@app.get("/api/status")
def get_status():
    return {
        "status"     : STATE["status"],
        "last_update": STATE["last_update"],
        "error"      : STATE["error"],
    }


@app.get("/api/prediction")
def get_prediction():
    if STATE["prediction"] is None:
        raise HTTPException(404, "No prediction yet — run the pipeline first")
    return STATE["prediction"]


@app.get("/api/backtest")
def get_backtest():
    return {
        "rule": STATE["backtest_rule"],
        "ml"  : STATE["backtest_ml"],
    }


@app.get("/api/trade")
def get_trade():
    return {"trade": STATE["trade"], "log": STATE["trade_log"]}


@app.post("/api/trade/close")
def close_trade(body: dict):
    t = STATE["trade"]
    if t is None or t["status"] != "open":
        raise HTTPException(400, "No open trade")
    close_px = body.get("close_price", t["entry"])
    reason   = body.get("reason", "manual")
    if t["direction"] == "LONG":
        pnl_pct = (close_px - t["entry"]) / t["entry"]
    else:
        pnl_pct = (t["entry"] - close_px) / t["entry"]
    pnl_usd = round(pnl_pct * t["notional_usd"], 2)

    t["status"]       = "closed"
    t["close_price"]  = close_px
    t["close_reason"] = reason
    t["pnl_pct"]      = round(pnl_pct * 100, 4)
    t["pnl_usd"]      = pnl_usd
    STATE["trade"]    = t
    return t


@app.get("/api/chart/dashboard")
def chart_dashboard():
    p = BASE / "strategy_dashboard.png"
    if not p.exists():
        raise HTTPException(404, "Chart not generated yet")
    return FileResponse(str(p), media_type="image/png")


@app.get("/api/chart/features")
def chart_features():
    p = BASE / "feature_importance.png"
    if not p.exists():
        raise HTTPException(404, "Chart not generated yet")
    return FileResponse(str(p), media_type="image/png")


@app.get("/api/health")
def health():
    return {"ok": True, "strategy_loaded": STRATEGY_LOADED}
