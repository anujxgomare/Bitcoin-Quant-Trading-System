# Bitcoin Quant Terminal — Setup Guide

## 📸 Dashboard Preview

<img width="975" height="461" alt="image" src="https://github.com/user-attachments/assets/6bdcf2fd-b8fb-4d98-8787-01921e6f929f" />
<img width="975" height="460" alt="image" src="https://github.com/user-attachments/assets/0e31af2f-2f37-42a4-8349-f02782c2ffb2" />

## 📊 Backtest Results

![Backtest](images/backtest.png)

## 🤖 Feature Importance

![Features](images/features.png)

## Files
```
trading-bot/
├── bitcoin_quant_strategy.py   ← strategy logic (already working)
├── server.py                   ← FastAPI backend
├── index.html                  ← frontend dashboard
├── requirements.txt            ← dependencies
└── README.md
```

## Step 1 — Install new dependencies
```bash
pip install fastapi uvicorn[standard] python-multipart
# (rest already installed from before)
```

## Step 2 — Run the server
```bash
# From your trading bot folder
uvicorn server:app --reload --port 8000
```

## Step 3 — Open the dashboard
Open browser → http://localhost:8000

## Step 4 — Run the strategy
Click **▶ RUN STRATEGY** button on the dashboard.
The pipeline runs in the background (~60-90 seconds).
When complete:
- Live prediction appears with probabilities
- Trade setup with SL / TP / BE / RR / position size
- Backtest results for both rule-based and ML strategies
- Click either chart image to expand fullscreen

## API Endpoints (for Jupyter or scripts)
```
POST /api/run          → trigger full pipeline
GET  /api/status       → check pipeline status
GET  /api/prediction   → latest bar prediction
GET  /api/backtest     → backtest metrics + equity
GET  /api/trade        → current demo trade + log
POST /api/trade/close  → close trade {close_price, reason}
GET  /api/chart/dashboard   → strategy_dashboard.png
GET  /api/chart/features    → feature_importance.png
```

## Notes
- The strategy module (bitcoin_quant_strategy.py) must be in the same folder as server.py
- Charts are saved as PNGs and served live — click to expand in the dashboard
- Demo trade is auto-generated from the latest bar prediction
- Position sizing uses 2% risk of $10,000 capital (edit CONFIG in bitcoin_quant_strategy.py)
