#!/usr/bin/env python3
"""Proxima X Command Center — High-Performance Cyberpunk Predictive Dashboard App."""

import os, sys, asyncio, webbrowser, json
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

sys.path.insert(0, str(Path(__file__).parent.parent))

from proxima_command_center.predictive_engine import PredictiveEngine
from proxima_command_center.mt5_history_loader import get_recent_mt5_executed_trades

app = FastAPI(title="Proxima X Command Center")

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

predictive_engine = PredictiveEngine()

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    with open(BASE_DIR / "templates" / "index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    lot_multiplier = 1.0 # Default $6k VPS Lot profile
    try:
        while True:
            # Check for client messages (lot size multiplier updates)
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                data = json.loads(msg)
                if "lot_multiplier" in data:
                    lot_multiplier = float(data["lot_multiplier"])
            except asyncio.TimeoutError:
                pass
            except Exception:
                pass

            predictions = predictive_engine.get_live_predictions(lot_multiplier=lot_multiplier)
            trades = get_recent_mt5_executed_trades(lot_multiplier=lot_multiplier)
            
            payload = {
                "predictions": predictions,
                "mt5_trades": trades,
                "lot_multiplier": lot_multiplier
            }
            await websocket.send_json(payload)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass

def main():
    import uvicorn
    port = 8888
    url = f"http://127.0.0.1:{port}"
    print("="*115)
    print("🚀 LAUNCHING PROXIMA X CYBERPUNK PREDICTIVE DASHBOARD COMMAND CENTER...")
    print(f"   URL: {url}")
    print("="*115)
    webbrowser.open(url)
    uvicorn.run("proxima_command_center.app:app", host="127.0.0.1", port=port, reload=False, log_level="error")

if __name__ == "__main__":
    main()
