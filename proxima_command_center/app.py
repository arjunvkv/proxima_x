#!/usr/bin/env python3
"""Proxima X Command Center — Multi-Page Predictive Institutional Terminal Flask Server."""

import sys, os, time, threading
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

from predictive_engine import PredictiveEngine
from mt5_history_loader import fetch_account_and_history
from vps_log_collector import fetch_vps_mt5_logs
from rolling_backtest_engine import RollingBacktestEngine

app = Flask(__name__)
app.config['SECRET_KEY'] = 'proxima_x_predictive_multipage_2026'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

engine = PredictiveEngine()
rolling_engine = RollingBacktestEngine()

# Compute Yesterday Full-Day Static Summary ONCE when the app starts
static_yesterday_summary = rolling_engine.get_yesterday_full_day_summary()

# Pre-run initial rolling backtest cycle
latest_rolling_backtest_data = rolling_engine.run_cycle()

@app.route('/')
def page_overview():
    return render_template('index.html', active_page='overview')

@app.route('/diagnostics')
def page_diagnostics():
    return render_template('diagnostics.html', active_page='diagnostics')

@app.route('/vps-logs')
def page_vps_logs():
    return render_template('vps_logs.html', active_page='vps-logs')

@app.route('/analytics')
def page_analytics():
    return render_template('analytics.html', active_page='analytics')

@app.route('/rolling-backtest')
def page_rolling_backtest():
    return render_template('rolling_backtest.html', active_page='rolling-backtest')

@app.route('/yesterday-summary')
def page_yesterday_summary():
    return render_template('yesterday_summary.html', active_page='yesterday-summary', summary=static_yesterday_summary)

@app.route('/api/yesterday_summary')
def api_yesterday_summary():
    return jsonify(static_yesterday_summary)

@app.route('/api/rolling_backtest')
def api_rolling_backtest():
    return jsonify(latest_rolling_backtest_data)

@app.route('/api/predictive_radar')
def api_predictive_radar():
    predictions, radar, diagnostics, exposure, imminent, config = engine.get_live_predictions()
    mt5_data = fetch_account_and_history()
    vps_logs = fetch_vps_mt5_logs()
    return jsonify({
        "predictions": predictions,
        "radar": radar,
        "diagnostics": diagnostics,
        "exposure": exposure,
        "imminent": imminent,
        "config": config,
        "mt5_telemetry": mt5_data,
        "vps_logs": vps_logs,
        "rolling_backtest": latest_rolling_backtest_data,
        "yesterday_summary": static_yesterday_summary
    })

def background_radar_broadcaster():
    """Broadcasts real-time predictive telemetry over SocketIO every 1 second."""
    global latest_rolling_backtest_data
    print("🟢 Proxima X Multi-Page Telemetry Broadcaster started (1s interval)...")
    last_5min_check = 0

    while True:
        try:
            now_time = time.time()
            if now_time - last_5min_check >= 300:
                latest_rolling_backtest_data = rolling_engine.run_cycle()
                last_5min_check = now_time

            predictions, radar, diagnostics, exposure, imminent, config = engine.get_live_predictions()
            mt5_data = fetch_account_and_history()
            vps_logs = fetch_vps_mt5_logs()
            
            socketio.emit('radar_update', {
                "predictions": predictions,
                "radar": radar,
                "diagnostics": diagnostics,
                "exposure": exposure,
                "imminent": imminent,
                "config": config,
                "mt5_telemetry": mt5_data,
                "vps_logs": vps_logs,
                "rolling_backtest": latest_rolling_backtest_data,
                "yesterday_summary": static_yesterday_summary,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            })
        except Exception as e:
            print("Broadcaster error:", e)
        time.sleep(1.0)

if __name__ == '__main__':
    t = threading.Thread(target=background_radar_broadcaster, daemon=True)
    t.start()
    print("=" * 115)
    print("PROXIMA X — MULTI-PAGE INSTITUTIONAL TERMINAL WEB SERVER RUNNING AT http://127.0.0.1:8888")
    print("=" * 115)
    socketio.run(app, host='0.0.0.0', port=8888, debug=False, allow_unsafe_werkzeug=True)
