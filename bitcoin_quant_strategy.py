# =============================================================================
#  BITCOIN QUANT TRADING STRATEGY
#  Composite of 5 strategies: EMA Trend + RSI Reversion + MACD Momentum +
#  Bollinger Band Squeeze + Volume Spike
#  Timeframe: 15-minute | Target accuracy: >=60%
# =============================================================================
#
#  INSTALL DEPENDENCIES:
#  pip install ccxt pandas numpy pandas-ta scikit-learn xgboost matplotlib
#       seaborn joblib python-dotenv
#
# =============================================================================

# ── 0. Imports ────────────────────────────────────────────────────────────────
import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import joblib
import os
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── 1. Configuration ──────────────────────────────────────────────────────────
CONFIG = {
    # Data
    "exchange"       : "binance",
    "symbol"         : "BTC/USDT",
    "timeframe"      : "15m",
    "limit"          : 5000,          # candles to fetch (~52 days of 15m bars)

    # EMA Trend
    "ema_fast"       : 21,
    "ema_slow"       : 55,

    # RSI
    "rsi_period"     : 14,
    "rsi_oversold"   : 35,
    "rsi_overbought" : 65,

    # MACD
    "macd_fast"      : 12,
    "macd_slow"      : 26,
    "macd_signal"    : 9,

    # Bollinger Bands
    "bb_period"      : 20,
    "bb_std"         : 2.0,

    # Volume spike
    "vol_period"     : 20,
    "vol_multiplier" : 1.5,

    # ATR for risk
    "atr_period"     : 14,

    # Signal weights (must sum to 1.0)
    "w_ema"          : 0.30,
    "w_rsi"          : 0.20,
    "w_macd"         : 0.20,
    "w_bb"           : 0.15,
    "w_vol"          : 0.15,

    # Trade filter
    "score_threshold": 0.50,

    # Risk management
    "atr_sl_mult"    : 1.5,
    "atr_tp_mult"    : 3.0,
    "risk_pct"       : 0.02,          # 2% of capital per trade
    "initial_capital": 10_000,

    # Forward window for labelling
    "label_fwd_bars" : 4,             # 4 bars = 1 hour ahead on 15m
    "label_min_move" : 0.002,         # 0.2% minimum move to label as directional

    # Model
    "model_path"     : "btc_model.pkl",
    "scaler_path"    : "btc_scaler.pkl",
}


# =============================================================================
# SECTION 1 — DATA COLLECTION
# =============================================================================

def fetch_ohlcv(cfg: dict) -> pd.DataFrame:
    """Fetch OHLCV data from Binance via ccxt (no API key needed for public data)."""
    print(f"\n{'='*60}")
    print(f"  Fetching {cfg['limit']} × {cfg['timeframe']} candles for {cfg['symbol']}")
    print(f"{'='*60}")

    exchange = getattr(ccxt, cfg["exchange"])({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })

    all_candles = []
    since = exchange.milliseconds() - cfg["limit"] * _tf_to_ms(cfg["timeframe"])

    while True:
        candles = exchange.fetch_ohlcv(
            cfg["symbol"], cfg["timeframe"],
            since=since, limit=1000
        )
        if not candles:
            break
        all_candles += candles
        since = candles[-1][0] + 1
        if len(all_candles) >= cfg["limit"]:
            break

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated()].sort_index()
    df = df.iloc[-cfg["limit"]:]          # keep exact limit

    print(f"  ✓ Fetched {len(df)} rows | {df.index[0]} → {df.index[-1]}")
    return df


def _tf_to_ms(tf: str) -> int:
    mapping = {"1m": 60_000, "5m": 300_000, "15m": 900_000,
               "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
    return mapping[tf]


# =============================================================================
# SECTION 2 — FEATURE ENGINEERING (5 Strategies)
# =============================================================================

def compute_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute all technical indicators for the 5 sub-strategies."""
    d = df.copy()

    # ── Strategy 1: EMA Trend ─────────────────────────────────────────────────
    d["ema_fast"] = ta.ema(d["close"], length=cfg["ema_fast"])
    d["ema_slow"] = ta.ema(d["close"], length=cfg["ema_slow"])
    # Signal: +1 bullish, -1 bearish, 0 neutral
    d["sig_ema"] = np.where(
        (d["ema_fast"] > d["ema_slow"]) & (d["close"] > d["ema_fast"]),  1,
        np.where(
        (d["ema_fast"] < d["ema_slow"]) & (d["close"] < d["ema_fast"]), -1, 0)
    )

    # ── Strategy 2: RSI Mean Reversion ───────────────────────────────────────
    d["rsi"] = ta.rsi(d["close"], length=cfg["rsi_period"])
    d["sig_rsi"] = np.where(d["rsi"] < cfg["rsi_oversold"],   1,
                   np.where(d["rsi"] > cfg["rsi_overbought"], -1, 0))

    # ── Strategy 3: MACD Momentum ────────────────────────────────────────────
    macd = ta.macd(d["close"],
                   fast=cfg["macd_fast"],
                   slow=cfg["macd_slow"],
                   signal=cfg["macd_signal"])
    d["macd_line"]   = macd.iloc[:, 0]
    d["macd_signal"] = macd.iloc[:, 1]
    d["macd_hist"]   = macd.iloc[:, 2]
    # Crossover: histogram turns positive or negative
    d["sig_macd"] = np.where(
        (d["macd_hist"] > 0) & (d["macd_hist"].shift(1) <= 0),  1,
        np.where(
        (d["macd_hist"] < 0) & (d["macd_hist"].shift(1) >= 0), -1,
        np.where(d["macd_hist"] > 0,  0.5,
        np.where(d["macd_hist"] < 0, -0.5, 0)))
    )

    # ── Strategy 4: Bollinger Band Squeeze ───────────────────────────────────
    bb = ta.bbands(d["close"], length=cfg["bb_period"], std=cfg["bb_std"])
    d["bb_upper"]  = bb.iloc[:, 0]
    d["bb_mid"]    = bb.iloc[:, 1]
    d["bb_lower"]  = bb.iloc[:, 2]
    d["bb_width"]  = (d["bb_upper"] - d["bb_lower"]) / d["bb_mid"]
    d["bb_width_avg"] = d["bb_width"].rolling(cfg["bb_period"]).mean()
    d["bb_squeeze"]   = d["bb_width"] < d["bb_width_avg"]   # True = squeeze
    # Signal: breakout direction after squeeze
    d["sig_bb"] = np.where(
        (~d["bb_squeeze"]) & d["bb_squeeze"].shift(1) & (d["close"] > d["bb_mid"]),  1,
        np.where(
        (~d["bb_squeeze"]) & d["bb_squeeze"].shift(1) & (d["close"] < d["bb_mid"]), -1,
        np.where(d["close"] > d["bb_upper"],  0.5,
        np.where(d["close"] < d["bb_lower"], -0.5, 0)))
    )

    # ── Strategy 5: Volume Spike ──────────────────────────────────────────────
    d["vol_avg"]   = d["volume"].rolling(cfg["vol_period"]).mean()
    d["vol_spike"] = d["volume"] > cfg["vol_multiplier"] * d["vol_avg"]
    # Direction from price action on spike bar
    d["bar_dir"]   = np.where(d["close"] >= d["open"], 1, -1)
    d["sig_vol"]   = np.where(d["vol_spike"], d["bar_dir"] * 1.0, 0.0)

    # ── ATR for risk management ───────────────────────────────────────────────
    d["atr"] = ta.atr(d["high"], d["low"], d["close"], length=cfg["atr_period"])

    # ── Composite Score ───────────────────────────────────────────────────────
    d["composite_score"] = (
        cfg["w_ema"]  * d["sig_ema"]  +
        cfg["w_rsi"]  * d["sig_rsi"]  +
        cfg["w_macd"] * d["sig_macd"] +
        cfg["w_bb"]   * d["sig_bb"]   +
        cfg["w_vol"]  * d["sig_vol"]
    )

    # Rule-based signal (used for backtesting & as a feature)
    d["rule_signal"] = np.where(
        d["composite_score"] >=  cfg["score_threshold"],  1,
        np.where(
        d["composite_score"] <= -cfg["score_threshold"], -1, 0)
    )

    print(f"  ✓ Indicators computed | {d.shape[0]} rows × {d.shape[1]} cols")
    return d


# =============================================================================
# SECTION 3 — FEATURE MATRIX & LABELLING
# =============================================================================

FEATURE_COLS = [
    "ema_fast", "ema_slow", "sig_ema",
    "rsi", "sig_rsi",
    "macd_line", "macd_signal", "macd_hist", "sig_macd",
    "bb_width", "bb_squeeze", "sig_bb",
    "vol_avg", "vol_spike", "sig_vol",
    "atr",
    "composite_score", "rule_signal",
    # Extra raw features
    "close", "high", "low", "open", "volume",
]


def label_data(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Create forward-looking labels:
      1  = LONG  (price rises ≥ label_min_move in next label_fwd_bars bars)
     -1  = SHORT (price falls ≥ label_min_move)
      0  = FLAT  (neither)
    For ML we map to 3-class: 0=SHORT, 1=FLAT, 2=LONG
    """
    fwd = cfg["label_fwd_bars"]
    min_move = cfg["label_min_move"]

    future_max = df["high"].shift(-fwd).rolling(fwd).max()
    future_min = df["low"].shift(-fwd).rolling(fwd).min()

    up_move   = (future_max - df["close"]) / df["close"]
    down_move = (df["close"] - future_min) / df["close"]

    df["label_raw"] = np.where(
        up_move   >= min_move,  1,
        np.where(
        down_move >= min_move, -1, 0)
    )
    df["label"] = df["label_raw"] + 1   # 0=SHORT, 1=FLAT, 2=LONG

    df.dropna(inplace=True)
    print(f"  ✓ Labels assigned | distribution: {df['label'].value_counts().to_dict()}")
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    y = df["label"].copy()
    # Cast bool columns
    for col in X.select_dtypes(include="bool").columns:
        X[col] = X[col].astype(int)
    return X, y


# =============================================================================
# SECTION 4 — MODEL TRAINING
# =============================================================================

def train_model(X: pd.DataFrame, y: pd.Series, cfg: dict):
    """
    Train an XGBoost classifier (best for structured financial time series).
    Uses TimeSeriesSplit to avoid look-ahead bias.
    Also trains Random Forest as ensemble backup.
    """
    print(f"\n{'='*60}")
    print("  Training models (XGBoost + Random Forest ensemble)")
    print(f"{'='*60}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=5)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb_scores = cross_val_score(xgb, X_scaled, y, cv=tscv, scoring="accuracy")
    print(f"  XGBoost CV accuracy: {xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}")

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1,
    )
    rf_scores = cross_val_score(rf, X_scaled, y, cv=tscv, scoring="accuracy")
    print(f"  Random Forest CV accuracy: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")

    # ── Final fit on full data ────────────────────────────────────────────────
    xgb.fit(X_scaled, y)
    rf.fit(X_scaled, y)

    # Blended ensemble (soft vote 60/40)
    class BlendedModel:
        def __init__(self, m1, m2, w1=0.6, w2=0.4):
            self.m1, self.m2, self.w1, self.w2 = m1, m2, w1, w2

        def predict_proba(self, X):
            return self.w1 * self.m1.predict_proba(X) + self.w2 * self.m2.predict_proba(X)

        def predict(self, X):
            return np.argmax(self.predict_proba(X), axis=1)

    model = BlendedModel(xgb, rf)

    # Train-set evaluation
    y_pred = model.predict(X_scaled)
    acc = accuracy_score(y, y_pred)
    print(f"\n  Train-set accuracy  : {acc:.4f}")
    print(f"\n  Classification report (train):")
    print(classification_report(y, y_pred, target_names=["SHORT", "FLAT", "LONG"]))

    # Save
    joblib.dump(scaler, cfg["scaler_path"])
    joblib.dump({"xgb": xgb, "rf": rf}, cfg["model_path"])
    print(f"  ✓ Model saved → {cfg['model_path']}")

    return model, scaler, xgb


# =============================================================================
# SECTION 5 — BACKTESTING (RULE-BASED + ML)
# =============================================================================

def backtest(df: pd.DataFrame, cfg: dict, use_ml: bool = False,
             model=None, scaler=None, X: pd.DataFrame = None) -> pd.DataFrame:
    """
    Vectorised backtest. Returns a trades DataFrame.
    use_ml=True → uses model predictions; False → uses rule_signal.
    """
    capital    = cfg["initial_capital"]
    risk_pct   = cfg["risk_pct"]
    sl_mult    = cfg["atr_sl_mult"]
    tp_mult    = cfg["atr_tp_mult"]

    trades = []
    in_trade = False
    entry_price = sl = tp = direction = 0

    if use_ml and model is not None:
        X_sc   = scaler.transform(X.loc[df.index])
        signals_raw = model.predict(X_sc)          # 0=SHORT, 1=FLAT, 2=LONG
        signal_map  = {0: -1, 1: 0, 2: 1}
        signals = pd.Series([signal_map[s] for s in signals_raw], index=df.index)
    else:
        signals = df["rule_signal"]

    for i, (ts, row) in enumerate(df.iterrows()):
        sig = signals.loc[ts]
        atr = row["atr"] if not np.isnan(row["atr"]) else 0

        if in_trade:
            # Check exit
            if direction == 1:
                if row["low"] <= sl:
                    pnl = (sl - entry_price) / entry_price
                    trades.append({"entry_time": entry_ts, "exit_time": ts,
                                   "direction": "LONG", "pnl_pct": pnl, "result": "SL"})
                    in_trade = False
                elif row["high"] >= tp:
                    pnl = (tp - entry_price) / entry_price
                    trades.append({"entry_time": entry_ts, "exit_time": ts,
                                   "direction": "LONG", "pnl_pct": pnl, "result": "TP"})
                    in_trade = False
            else:  # SHORT
                if row["high"] >= sl:
                    pnl = (entry_price - sl) / entry_price
                    trades.append({"entry_time": entry_ts, "exit_time": ts,
                                   "direction": "SHORT", "pnl_pct": -abs(pnl), "result": "SL"})
                    in_trade = False
                elif row["low"] <= tp:
                    pnl = (entry_price - tp) / entry_price
                    trades.append({"entry_time": entry_ts, "exit_time": ts,
                                   "direction": "SHORT", "pnl_pct": pnl, "result": "TP"})
                    in_trade = False

        if not in_trade and sig != 0 and atr > 0:
            direction = sig
            entry_price = row["close"]
            entry_ts    = ts
            if direction == 1:
                sl = entry_price - sl_mult * atr
                tp = entry_price + tp_mult * atr
            else:
                sl = entry_price + sl_mult * atr
                tp = entry_price - tp_mult * atr
            in_trade = True

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        print("  No trades generated.")
        return trades_df

    trades_df["win"]      = trades_df["pnl_pct"] > 0
    trades_df["capital"]  = cfg["initial_capital"] * (1 + trades_df["pnl_pct"].cumsum())

    label = "ML" if use_ml else "Rule"
    win_rate = trades_df["win"].mean()
    total_ret = trades_df["pnl_pct"].sum()

    print(f"\n  [{label}] Backtest Results")
    print(f"  {'─'*40}")
    print(f"  Total trades   : {len(trades_df)}")
    print(f"  Win rate       : {win_rate:.2%}")
    print(f"  Total return   : {total_ret:.2%}")
    print(f"  Avg trade PnL  : {trades_df['pnl_pct'].mean():.4%}")
    print(f"  Best trade     : {trades_df['pnl_pct'].max():.4%}")
    print(f"  Worst trade    : {trades_df['pnl_pct'].min():.4%}")
    tp_count = (trades_df["result"] == "TP").sum()
    sl_count = (trades_df["result"] == "SL").sum()
    print(f"  TP / SL hits   : {tp_count} / {sl_count}")

    return trades_df


# =============================================================================
# SECTION 6 — PREDICTION (LIVE / LATEST BAR)
# =============================================================================

def predict_latest(df: pd.DataFrame, model, scaler, cfg: dict) -> dict:
    """
    Generate a prediction for the most recent complete bar.
    Returns a dict with signal, score, entry, SL, TP levels.
    """
    X, _ = build_feature_matrix(df)
    latest_X = X.iloc[[-1]]
    latest_row = df.iloc[-1]

    X_sc   = scaler.transform(latest_X)
    proba  = model.predict_proba(X_sc)[0]   # [P(SHORT), P(FLAT), P(LONG)]
    pred   = int(np.argmax(proba))
    label_map = {0: "SHORT", 1: "FLAT", 2: "LONG"}

    score   = latest_row["composite_score"]
    atr     = latest_row["atr"]
    close   = latest_row["close"]

    if pred == 2:   # LONG
        sl = close - cfg["atr_sl_mult"] * atr
        tp = close + cfg["atr_tp_mult"] * atr
    elif pred == 0: # SHORT
        sl = close + cfg["atr_sl_mult"] * atr
        tp = close - cfg["atr_tp_mult"] * atr
    else:
        sl = tp = None

    result = {
        "timestamp"       : df.index[-1],
        "close"           : round(close, 2),
        "prediction"      : label_map[pred],
        "confidence"      : round(float(proba[pred]), 4),
        "p_long"          : round(float(proba[2]), 4),
        "p_flat"          : round(float(proba[1]), 4),
        "p_short"         : round(float(proba[0]), 4),
        "composite_score" : round(float(score), 4),
        "rule_signal"     : int(latest_row["rule_signal"]),
        "atr"             : round(float(atr), 2),
        "stop_loss"       : round(sl, 2) if sl else None,
        "take_profit"     : round(tp, 2) if tp else None,
        "rsi"             : round(float(latest_row["rsi"]), 2),
        "macd_hist"       : round(float(latest_row["macd_hist"]), 4),
        "bb_squeeze"      : bool(latest_row["bb_squeeze"]),
        "vol_spike"       : bool(latest_row["vol_spike"]),
    }
    return result


# =============================================================================
# SECTION 7 — VISUALISATION
# =============================================================================

def plot_strategy_dashboard(df: pd.DataFrame, trades_df: pd.DataFrame, cfg: dict):
    """Plot a 4-panel strategy dashboard."""
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

    clr_bg   = "#0f1117"
    clr_text = "#e0e0e0"
    clr_up   = "#26a69a"
    clr_dn   = "#ef5350"
    clr_neu  = "#78909c"

    tail = df.tail(500)

    # ── Panel 1: Price + EMAs + BB ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(clr_bg)
    ax1.plot(tail.index, tail["close"],    color="#b0bec5", linewidth=0.8, label="Close")
    ax1.plot(tail.index, tail["ema_fast"], color="#ffa726", linewidth=1.2, label=f"EMA{cfg['ema_fast']}")
    ax1.plot(tail.index, tail["ema_slow"], color="#42a5f5", linewidth=1.2, label=f"EMA{cfg['ema_slow']}")
    ax1.fill_between(tail.index, tail["bb_upper"], tail["bb_lower"],
                     alpha=0.08, color="#7e57c2")
    ax1.plot(tail.index, tail["bb_upper"], color="#7e57c2", linewidth=0.5, linestyle="--")
    ax1.plot(tail.index, tail["bb_lower"], color="#7e57c2", linewidth=0.5, linestyle="--")
    # Plot trade entries
    if not trades_df.empty:
        longs  = trades_df[trades_df["direction"] == "LONG"]
        shorts = trades_df[trades_df["direction"] == "SHORT"]
        for _, t in longs.iterrows():
            if t["entry_time"] in tail.index:
                ax1.axvline(t["entry_time"], color=clr_up, alpha=0.3, linewidth=0.7)
        for _, t in shorts.iterrows():
            if t["entry_time"] in tail.index:
                ax1.axvline(t["entry_time"], color=clr_dn, alpha=0.3, linewidth=0.7)
    ax1.set_title("BTC/USDT 15m — Price · EMAs · Bollinger Bands",
                  color=clr_text, fontsize=11)
    ax1.legend(loc="upper left", fontsize=8, facecolor=clr_bg, labelcolor=clr_text)
    ax1.tick_params(colors=clr_text)
    ax1.spines[:].set_color("#2a2d3a")

    # ── Panel 2: RSI ──────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(clr_bg)
    ax2.plot(tail.index, tail["rsi"], color="#ce93d8", linewidth=0.9)
    ax2.axhline(cfg["rsi_oversold"],   color=clr_up, linewidth=0.8, linestyle="--")
    ax2.axhline(cfg["rsi_overbought"], color=clr_dn, linewidth=0.8, linestyle="--")
    ax2.axhline(50, color=clr_neu, linewidth=0.5, linestyle=":")
    ax2.fill_between(tail.index, tail["rsi"], cfg["rsi_oversold"],
                     where=tail["rsi"] < cfg["rsi_oversold"], alpha=0.3, color=clr_up)
    ax2.fill_between(tail.index, tail["rsi"], cfg["rsi_overbought"],
                     where=tail["rsi"] > cfg["rsi_overbought"], alpha=0.3, color=clr_dn)
    ax2.set_ylim(0, 100)
    ax2.set_title(f"RSI ({cfg['rsi_period']})", color=clr_text, fontsize=10)
    ax2.tick_params(colors=clr_text); ax2.spines[:].set_color("#2a2d3a")

    # ── Panel 3: MACD ─────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(clr_bg)
    colors_hist = [clr_up if v >= 0 else clr_dn for v in tail["macd_hist"]]
    ax3.bar(tail.index, tail["macd_hist"], color=colors_hist, alpha=0.7, width=0.004)
    ax3.plot(tail.index, tail["macd_line"],   color="#ffa726", linewidth=0.9, label="MACD")
    ax3.plot(tail.index, tail["macd_signal"], color="#ef5350", linewidth=0.9, label="Signal")
    ax3.axhline(0, color=clr_neu, linewidth=0.5)
    ax3.set_title(f"MACD ({cfg['macd_fast']}/{cfg['macd_slow']}/{cfg['macd_signal']})",
                  color=clr_text, fontsize=10)
    ax3.legend(loc="upper left", fontsize=7, facecolor=clr_bg, labelcolor=clr_text)
    ax3.tick_params(colors=clr_text); ax3.spines[:].set_color("#2a2d3a")

    # ── Panel 4: Composite Score ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    ax4.set_facecolor(clr_bg)
    score_colors = [clr_up if s >= cfg["score_threshold"]
                    else (clr_dn if s <= -cfg["score_threshold"] else clr_neu)
                    for s in tail["composite_score"]]
    ax4.bar(tail.index, tail["composite_score"], color=score_colors, alpha=0.8, width=0.006)
    ax4.axhline( cfg["score_threshold"], color=clr_up, linewidth=1, linestyle="--")
    ax4.axhline(-cfg["score_threshold"], color=clr_dn, linewidth=1, linestyle="--")
    ax4.axhline(0, color=clr_neu, linewidth=0.5)
    ax4.set_title("Composite Score (5-strategy weighted sum)", color=clr_text, fontsize=10)
    ax4.set_ylim(-1.1, 1.1)
    ax4.tick_params(colors=clr_text); ax4.spines[:].set_color("#2a2d3a")

    # ── Panel 5: Equity Curve ─────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, :])
    ax5.set_facecolor(clr_bg)
    if not trades_df.empty:
        equity = [cfg["initial_capital"]]
        for _, t in trades_df.iterrows():
            equity.append(equity[-1] * (1 + t["pnl_pct"]))
        ax5.plot(range(len(equity)), equity, color="#26a69a", linewidth=1.2)
        ax5.fill_between(range(len(equity)), equity, cfg["initial_capital"],
                         where=[e >= cfg["initial_capital"] for e in equity],
                         alpha=0.2, color=clr_up)
        ax5.fill_between(range(len(equity)), equity, cfg["initial_capital"],
                         where=[e < cfg["initial_capital"] for e in equity],
                         alpha=0.2, color=clr_dn)
        ax5.axhline(cfg["initial_capital"], color=clr_neu, linewidth=0.8, linestyle="--")
        win_rate = trades_df["win"].mean()
        final = equity[-1]
        ax5.set_title(
            f"Equity Curve — {len(trades_df)} trades | Win rate: {win_rate:.1%} | "
            f"Final: ${final:,.0f} ({(final/cfg['initial_capital']-1)*100:.1f}%)",
            color=clr_text, fontsize=10
        )
    ax5.tick_params(colors=clr_text); ax5.spines[:].set_color("#2a2d3a")

    plt.suptitle("Bitcoin 15m Composite Strategy — Full Dashboard",
                 color=clr_text, fontsize=14, y=1.01)
    plt.savefig("strategy_dashboard.png", dpi=150, bbox_inches="tight",
                facecolor=clr_bg)
    plt.show()
    print("\n  ✓ Dashboard saved → strategy_dashboard.png")


def plot_feature_importance(xgb_model, feature_names: list):
    """Plot XGBoost feature importances."""
    imp = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(8, 10))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    colors = ["#26a69a" if v > imp.median() else "#78909c" for v in imp]
    imp.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("XGBoost Feature Importance", color="#e0e0e0", fontsize=12)
    ax.tick_params(colors="#e0e0e0")
    ax.spines[:].set_color("#2a2d3a")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight",
                facecolor="#0f1117")
    plt.show()
    print("  ✓ Feature importance saved → feature_importance.png")


# =============================================================================
# SECTION 8 — MAIN PIPELINE
# =============================================================================

def main():
    print("\n" + "█"*60)
    print("  BITCOIN 15m COMPOSITE QUANT STRATEGY")
    print("█"*60)

    # 1. Fetch data
    df_raw = fetch_ohlcv(CONFIG)

    # 2. Compute indicators
    print(f"\n{'='*60}")
    print("  Computing indicators & signals")
    print(f"{'='*60}")
    df = compute_indicators(df_raw, CONFIG)

    # 3. Label data
    print(f"\n{'='*60}")
    print("  Generating forward-looking labels")
    print(f"{'='*60}")
    df = label_data(df, CONFIG)

    # 4. Feature matrix
    X, y = build_feature_matrix(df)
    print(f"\n  Feature matrix shape: {X.shape}")
    print(f"  Label counts: {y.value_counts().to_dict()} (0=SHORT, 1=FLAT, 2=LONG)")

    # 5. Train model
    model, scaler, xgb_model = train_model(X, y, CONFIG)

    # 6. Backtest — Rule-based
    print(f"\n{'='*60}")
    print("  Backtesting (Rule-Based Signal)")
    print(f"{'='*60}")
    trades_rule = backtest(df, CONFIG, use_ml=False)

    # 7. Backtest — ML
    print(f"\n{'='*60}")
    print("  Backtesting (ML Model)")
    print(f"{'='*60}")
    trades_ml = backtest(df, CONFIG, use_ml=True,
                         model=model, scaler=scaler, X=X)

    # 8. Live prediction
    print(f"\n{'='*60}")
    print("  LIVE PREDICTION — Latest bar")
    print(f"{'='*60}")
    pred = predict_latest(df, model, scaler, CONFIG)
    print(f"\n  Timestamp       : {pred['timestamp']}")
    print(f"  BTC/USDT Close  : ${pred['close']:,}")
    print(f"  ── Signal ──────────────────────────────")
    print(f"  Prediction      : {pred['prediction']}  (confidence {pred['confidence']:.1%})")
    print(f"  P(LONG)         : {pred['p_long']:.1%}")
    print(f"  P(FLAT)         : {pred['p_flat']:.1%}")
    print(f"  P(SHORT)        : {pred['p_short']:.1%}")
    print(f"  Composite score : {pred['composite_score']}")
    print(f"  Rule signal     : {pred['rule_signal']}")
    print(f"  ── Indicators ──────────────────────────")
    print(f"  RSI             : {pred['rsi']}")
    print(f"  MACD histogram  : {pred['macd_hist']}")
    print(f"  BB squeeze      : {pred['bb_squeeze']}")
    print(f"  Volume spike    : {pred['vol_spike']}")
    print(f"  ATR             : {pred['atr']}")
    print(f"  ── Risk Levels ─────────────────────────")
    print(f"  Stop Loss       : ${pred['stop_loss']}")
    print(f"  Take Profit     : ${pred['take_profit']}")

    # 9. Plots
    print(f"\n{'='*60}")
    print("  Generating charts")
    print(f"{'='*60}")
    plot_strategy_dashboard(df, trades_rule, CONFIG)
    plot_feature_importance(xgb_model, X.columns.tolist())

    return df, model, scaler, trades_rule, trades_ml, pred


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    df, model, scaler, trades_rule, trades_ml, pred = main()
