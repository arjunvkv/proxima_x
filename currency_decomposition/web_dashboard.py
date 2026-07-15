"""
web_dashboard.py — Proxima Currency Decomposition Engine — Game-Level Web Dashboard
======================================================================================
Runs as a COMPLETELY SEPARATE process — zero performance impact on the trading engine.

Reads:  logs/dashboard_log.jsonl   (written every 30s by the engine — already exists)
Method: tail-read (seek to end, read last chunk) — O(1), never scans full file

Usage:
    cd currency_decomposition
    python web_dashboard.py            # http://localhost:7700
    python web_dashboard.py 8080       # custom port

DO NOT import this from main.py or runtime/manager.py.
"""
import json
import os
import sys
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
PORT        = int(sys.argv[1]) if len(sys.argv) > 1 else 7700
_HERE       = Path(__file__).parent
LATEST_FILE = _HERE / "logs" / "dashboard_latest.json"
POLL_SEC    = 0.5   # 500ms poll interval — instant updates, zero CPU or IO footprint

# ── Global state — written by poller thread, read by HTTP handler ─────────────
_state: dict = {}
_state_lock = threading.Lock()
_state_ts: float = 0.0          # epoch when state was last updated
_sse_clients: list = []          # open SSE response objects
_sse_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
#  File reader — reads latest JSON file directly
# ─────────────────────────────────────────────────────────────────────────────
def _read_latest_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
#  Background poller — runs in daemon thread, never touches engine state
# ─────────────────────────────────────────────────────────────────────────────
def _poller_loop():
    global _state, _state_ts
    while True:
        try:
            data = _read_latest_json(LATEST_FILE)
            if data:
                # Check if the engine is actively writing to the log file (within last 60s)
                mtime = LATEST_FILE.stat().st_mtime if LATEST_FILE.exists() else 0
                engine_active = (time.time() - mtime) < 60.0
                data["engine_active"] = engine_active

                payload = json.dumps(data)
                with _state_lock:
                    _state = data
                    _state_ts = time.time()
                # push to all open SSE clients
                msg = f"data: {payload}\n\n"
                with _sse_lock:
                    dead = []
                    for wfile in _sse_clients:
                        try:
                            wfile.write(msg.encode())
                            wfile.flush()
                        except Exception:
                            dead.append(wfile)
                    for d in dead:
                        _sse_clients.remove(d)
        except Exception:
            pass
        time.sleep(POLL_SEC)


# ─────────────────────────────────────────────────────────────────────────────
#  HTML — the entire game-level UI as a single string
# ─────────────────────────────────────────────────────────────────────────────
_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>PROXIMA CDE — Live Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet"/>
<style>
:root{
  --bg:#05080f;--bg2:#080d18;--bg3:#0c1422;--bg4:#111c2e;
  --border:#1a2d4a;--border2:#1e3558;
  --cyan:#00e5ff;--cyan2:#00b8d4;--cyan3:#006080;
  --green:#00ff88;--green2:#00c96b;--green3:#003d20;
  --amber:#ffaa00;--amber2:#cc8800;--amber3:#3d2800;
  --red:#ff3b6b;--red2:#cc2d55;--red3:#3d0010;
  --purple:#b048ff;--purple2:#8533cc;--purple3:#2a0050;
  --text:#c8d8ee;--text2:#8aaac8;--text3:#4a6580;
  --font:'Inter',sans-serif;--mono:'JetBrains Mono',monospace;
  --r:8px;--r2:12px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--font);font-size:13px;min-height:100vh;overflow-x:hidden}

/* ── Scanline overlay ── */
body::before{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,229,255,.012) 2px,rgba(0,229,255,.012) 4px);pointer-events:none;z-index:9999}

/* ── Header ── */
#header{display:flex;align-items:center;gap:12px;padding:6px 16px;background:linear-gradient(90deg,rgba(0,229,255,.08),rgba(0,229,255,.02));border-bottom:1px solid var(--border2);position:sticky;top:0;z-index:100;backdrop-filter:blur(12px);height:42px}
.hdr-logo{font-family:var(--mono);font-size:13px;font-weight:700;color:var(--cyan);letter-spacing:1.5px;text-shadow:0 0 20px var(--cyan)}
.hdr-sep{color:var(--border2);font-size:16px}
.hdr-pill{padding:2px 8px;border-radius:20px;font-family:var(--mono);font-size:10px;font-weight:600;letter-spacing:.3px}
.pill-mode{background:rgba(0,229,255,.12);border:1px solid var(--cyan3);color:var(--cyan)}
.pill-ok{background:rgba(0,255,136,.1);border:1px solid var(--green3);color:var(--green)}
.pill-warn{background:rgba(255,170,0,.1);border:1px solid var(--amber3);color:var(--amber)}
.pill-err{background:rgba(255,59,107,.1);border:1px solid var(--red3);color:var(--red)}
.pill-neu{background:rgba(255,170,0,.12);border:1px solid rgba(255,170,0,.35);color:var(--amber)}
.hdr-right{margin-left:auto;display:flex;align-items:center;gap:10px}
#hdr-uptime,#hdr-cycle,#hdr-ts{font-family:var(--mono);font-size:10px;color:var(--text2)}
.pulse{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.8)}}

/* ── Navigation Links ── */
.hdr-nav{display:flex;gap:12px;margin:0 auto}
.nav-link{background:none;border:1px solid transparent;color:var(--text3);font-family:var(--mono);font-size:10px;font-weight:700;letter-spacing:1px;cursor:pointer;padding:4px 10px;border-radius:4px;transition:color .15s,background .15s,border-color .15s}
.nav-link:hover{color:var(--cyan);background:rgba(0,229,255,.05)}
.nav-link.active{color:var(--cyan);background:rgba(0,229,255,.1);border-color:rgba(0,229,255,.25)}

/* ── Grid layout ── */
#grid{display:grid;grid-template-columns:280px 1fr 260px;grid-template-rows:auto auto;gap:12px;padding:12px 14px;min-height:calc(100vh - 52px)}

/* ── Panels ── */
.panel{background:var(--bg2);border:1px solid var(--border);border-radius:var(--r2);padding:14px;position:relative;overflow:hidden}
.panel::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,rgba(0,229,255,.025) 0%,transparent 60%);pointer-events:none}
.panel-title{font-family:var(--mono);font-size:10px;font-weight:700;letter-spacing:2px;color:var(--text3);text-transform:uppercase;margin-bottom:12px;display:flex;align-items:center;gap:8px}
.panel-title .title-dot{width:5px;height:5px;border-radius:50%;background:var(--cyan);box-shadow:0 0 6px var(--cyan)}

/* ── Currency bars ── */
.ccy-row{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.ccy-label{font-family:var(--mono);font-size:12px;font-weight:700;width:28px;color:var(--text)}
.ccy-bar-wrap{flex:1;height:6px;background:var(--bg4);border-radius:3px;overflow:hidden;position:relative}
.ccy-bar{height:100%;border-radius:3px;transition:width .6s cubic-bezier(.4,0,.2,1);position:relative}
.ccy-bar::after{content:'';position:absolute;right:0;top:0;bottom:0;width:3px;border-radius:3px;filter:blur(2px)}
.ccy-bar.pos{background:linear-gradient(90deg,rgba(0,255,136,.3),var(--green))}
.ccy-bar.pos::after{background:var(--green)}
.ccy-bar.neg{background:linear-gradient(90deg,rgba(255,59,107,.3),var(--red))}
.ccy-bar.neg::after{background:var(--red)}
.ccy-bar.neu{background:var(--bg4)}
.ccy-val{font-family:var(--mono);font-size:11px;width:80px;text-align:right}
.ccy-meta{font-family:var(--mono);font-size:10px;color:var(--text3);width:40px;text-align:center}
.rank-top .ccy-label,.rank-top .ccy-val{color:var(--green);text-shadow:0 0 8px rgba(0,255,136,.4)}
.rank-bot .ccy-label,.rank-bot .ccy-val{color:var(--red);text-shadow:0 0 8px rgba(255,59,107,.4)}

/* ── Section divider ── */
.divider{height:1px;background:linear-gradient(90deg,transparent,var(--border),transparent);margin:10px 0}

/* ── Pipeline funnel ── */
#panel-pipeline{grid-column:2;grid-row:1/3}
.funnel-wrap{display:flex;flex-direction:column;gap:6px}
.funnel-stage{display:flex;align-items:center;gap:10px}
.funnel-label{font-family:var(--mono);font-size:10px;color:var(--text3);width:80px;letter-spacing:.5px}
.funnel-bar-wrap{flex:1;height:28px;background:var(--bg4);border-radius:4px;overflow:hidden;position:relative}
.funnel-bar{height:100%;border-radius:4px;display:flex;align-items:center;padding-left:8px;font-family:var(--mono);font-size:12px;font-weight:700;transition:width .8s cubic-bezier(.4,0,.2,1),background .4s;white-space:nowrap;position:relative;overflow:hidden}
.funnel-bar::before{content:'';position:absolute;inset:0;background:linear-gradient(90deg,rgba(255,255,255,.05),transparent)}
.funnel-arrow{color:var(--text3);font-size:14px}
.f-gen{background:linear-gradient(90deg,rgba(176,72,255,.4),var(--purple))}
.f-burst{background:linear-gradient(90deg,rgba(0,229,255,.4),var(--cyan))}
.f-bar{background:linear-gradient(90deg,rgba(0,184,212,.4),var(--cyan2))}
.f-ranked{background:linear-gradient(90deg,rgba(255,170,0,.4),var(--amber))}
.f-selected{background:linear-gradient(90deg,rgba(0,201,107,.4),var(--green2))}
.f-risk{background:linear-gradient(90deg,rgba(0,255,136,.4),var(--green))}
.f-exe{background:linear-gradient(90deg,rgba(0,255,136,.8),#00ff88);color:#000}

/* ── Graph gauge ── */
.gauge-wrap{display:flex;align-items:center;justify-content:center;gap:24px;padding:8px 0}
.gauge{width:90px;height:90px;position:relative}
.gauge svg{transform:rotate(-90deg)}
.gauge-bg{stroke:var(--bg4);fill:none}
.gauge-fill{fill:none;stroke-linecap:round;transition:stroke-dashoffset .8s cubic-bezier(.4,0,.2,1),stroke .4s}
.gauge-label{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;font-family:var(--mono)}
.gauge-val{font-size:18px;font-weight:700}
.gauge-sub{font-size:9px;color:var(--text3);letter-spacing:1px;margin-top:1px}
.gauge-legend{display:flex;flex-direction:column;gap:4px}
.gauge-leg-row{display:flex;align-items:center;gap:6px;font-family:var(--mono);font-size:10px;color:var(--text2)}
.gauge-leg-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}

/* ── Hypothesis card ── */
.hyp-card{background:var(--bg3);border:1px solid var(--border2);border-radius:var(--r);padding:12px;margin-top:8px}
.hyp-sym{font-family:var(--mono);font-size:22px;font-weight:700;color:var(--cyan);letter-spacing:2px;text-shadow:0 0 16px rgba(0,229,255,.5)}
.hyp-dir{font-family:var(--mono);font-size:13px;font-weight:700;margin-left:8px}
.hyp-dir.buy{color:var(--green)}
.hyp-dir.sell{color:var(--red)}
.hyp-row{display:flex;gap:16px;margin-top:8px}
.hyp-metric{display:flex;flex-direction:column;gap:2px}
.hyp-mlabel{font-size:9px;color:var(--text3);letter-spacing:1px;text-transform:uppercase}
.hyp-mval{font-family:var(--mono);font-size:14px;font-weight:700}

/* ── Positions ── */
.pos-card{background:var(--bg3);border:1px solid var(--border2);border-radius:var(--r);padding:10px 12px;margin-bottom:6px;transition:border-color .3s}
.pos-card:hover{border-color:var(--cyan3)}
.pos-header{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.pos-sym{font-family:var(--mono);font-size:13px;font-weight:700;color:var(--text)}
.pos-dir{font-family:var(--mono);font-size:10px;font-weight:700;padding:2px 7px;border-radius:10px}
.pos-dir.buy{background:rgba(0,255,136,.15);color:var(--green);border:1px solid rgba(0,255,136,.3)}
.pos-dir.sell{background:rgba(255,59,107,.15);color:var(--red);border:1px solid rgba(255,59,107,.3)}
.pos-pnl{margin-left:auto;font-family:var(--mono);font-size:13px;font-weight:700}
.pos-pnl.pos{color:var(--green)}
.pos-pnl.neg{color:var(--red)}
.pos-detail{font-family:var(--mono);font-size:10px;color:var(--text3);display:flex;gap:12px}
.no-pos{color:var(--text3);font-family:var(--mono);font-size:12px;text-align:center;padding:20px;border:1px dashed var(--border);border-radius:var(--r)}

/* ── Status bar ── */
#statusbar{grid-column:1/-1;display:flex;align-items:center;gap:8px;padding:8px 14px;background:var(--bg2);border:1px solid var(--border);border-radius:var(--r);flex-wrap:wrap}
.sb-chip{display:flex;align-items:center;gap:5px;padding:4px 10px;border-radius:20px;font-family:var(--mono);font-size:10px;font-weight:600;letter-spacing:.3px}
.sb-chip .sb-label{color:var(--text3)}
.sb-chip .sb-val{font-weight:700}
.chip-ok{background:rgba(0,255,136,.06);border:1px solid rgba(0,255,136,.2)}
.chip-ok .sb-val{color:var(--green)}
.chip-warn{background:rgba(255,170,0,.06);border:1px solid rgba(255,170,0,.2)}
.chip-warn .sb-val{color:var(--amber)}
.chip-err{background:rgba(255,59,107,.06);border:1px solid rgba(255,59,107,.2)}
.chip-err .sb-val{color:var(--red)}
.chip-neu{background:rgba(0,229,255,.04);border:1px solid rgba(0,229,255,.12)}
.chip-neu .sb-val{color:var(--cyan)}
.sb-sep{color:var(--border2);font-size:16px;margin:0 2px}

/* ── Universe ── */
.uni-row{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:11px;margin-bottom:6px}
.uni-val{font-size:15px;font-weight:700}
.uni-bar-wrap{flex:1;height:4px;background:var(--bg4);border-radius:2px;overflow:hidden}
.uni-bar{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--cyan2),var(--cyan));transition:width .6s}

/* ── Risk ── */
.risk-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.risk-label{font-size:10px;color:var(--text3);letter-spacing:.5px}
.risk-val{font-family:var(--mono);font-size:12px;font-weight:700;color:var(--text)}
.lots-bar-wrap{width:100%;height:6px;background:var(--bg4);border-radius:3px;overflow:hidden;margin-top:3px;margin-bottom:8px}
.lots-bar{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--cyan2),var(--cyan));transition:width .6s}
.cooldown-badge{background:var(--amber3);border:1px solid var(--amber2);border-radius:6px;padding:4px 10px;font-family:var(--mono);font-size:11px;font-weight:700;color:var(--amber);margin-top:4px;text-align:center}

/* Swing overlay table */
#swing-table tbody tr{border-bottom:1px solid var(--border);transition:background .15s}
#swing-table tbody tr:hover{background:var(--bg3)}
#swing-table tbody td{padding:4px 10px;white-space:nowrap}
#swing-table tbody td:first-child{font-weight:700;color:var(--text);font-size:12px}
.swing-bar-wrap{display:flex;align-items:center;gap:3px;width:100px;margin:0 auto}
.swing-bar{height:6px;border-radius:3px;position:relative;background:var(--bg4);overflow:hidden}
.swing-fill{position:absolute;top:0;height:100%;border-radius:3px;transition:left .4s,width .4s}
.swing-fill.pos{background:var(--green);left:60%;width:30%}
.swing-fill.neg{background:var(--red);left:10%;width:30%}
.swing-fill.mid{background:var(--cyan);left:40%;width:20%}
.swing-dot{position:absolute;top:-1px;width:8px;height:8px;border-radius:50%;background:#fff;border:2px solid var(--bg);z-index:2;transition:left .4s}

/* Swing reach progress bar */
.swing-reach-bar-wrap{display:inline-flex;align-items:center;gap:5px;width:100%;min-width:90px}
.swing-reach-bar{flex:1;height:7px;background:var(--bg4);border-radius:4px;overflow:hidden;position:relative}
.swing-reach-fill{height:100%;border-radius:4px;transition:width .4s}
.reach-low{background:var(--cyan2)}
.reach-mid{background:var(--amber)}
.reach-high{background:var(--green)}
.reach-over{background:var(--red)}
@keyframes reach-pulse{0%,100%{opacity:1}50%{opacity:.5}}
.reach-over{animation:reach-pulse 1s infinite}

/* Highlight borders of active trade rows */
#swing-table tbody tr.active-trade-row td {
  border-top: 1.5px solid var(--amber) !important;
  border-bottom: 1.5px solid var(--amber) !important;
  background: rgba(255, 170, 0, 0.04) !important;
}
#swing-table tbody tr.active-trade-row td:first-child {
  border-left: 1.5px solid var(--amber) !important;
}
#swing-table tbody tr.active-trade-row td:last-child {
  border-right: 1.5px solid var(--amber) !important;
}

/* ── Animations ── */
@keyframes fadein{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.fade-in{animation:fadein .4s ease}
@keyframes glow{0%,100%{box-shadow:0 0 0 rgba(0,229,255,0)}50%{box-shadow:0 0 20px rgba(0,229,255,.15)}}
.panel:hover{animation:glow 2s infinite}

@keyframes flash-glow{
  0%{border-color:var(--amber);box-shadow:0 0 16px rgba(255,170,0,0.4)}
  50%{border-color:var(--amber);box-shadow:0 0 24px rgba(255,170,0,0.6)}
  100%{border-color:var(--border);box-shadow:none}
}
.highlight-flash{animation:flash-glow 1.5s ease-out}

/* ── No-data overlay ── */
#nodata{position:fixed;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;background:var(--bg);z-index:200}
#nodata.hidden{display:none}
.nodata-icon{font-size:48px;margin-bottom:16px;opacity:.4}
.nodata-text{font-family:var(--mono);font-size:14px;color:var(--text3);letter-spacing:1px}
.nodata-path{font-family:var(--mono);font-size:11px;color:var(--text3);margin-top:8px;opacity:.6}
.nodata-spin{width:32px;height:32px;border:2px solid var(--border2);border-top-color:var(--cyan);border-radius:50%;animation:spin 1s linear infinite;margin-top:20px}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px}
</style>
</head>
<body>

<!-- No-data overlay -->
<div id="nodata">
  <div class="nodata-icon">⬡</div>
  <div class="nodata-text">AWAITING ENGINE DATA</div>
  <div class="nodata-path">logs/dashboard_log.jsonl</div>
  <div class="nodata-spin"></div>
</div>

<!-- Header -->
<div id="header">
  <div class="hdr-logo">⬡ PROXIMA CDE</div>
  <div class="hdr-sep">│</div>
  <div id="hdr-mode" class="hdr-pill pill-mode">LIVE</div>
  <div id="hdr-health" class="hdr-pill pill-ok">● OK</div>
  <div class="hdr-sep">│</div>
  
  <div class="hdr-nav">
    <button id="nav-overview" class="nav-link active">OVERVIEW</button>
    <button id="nav-swing" class="nav-link">SWING ANALYTICS</button>
  </div>

  <div class="hdr-right">
    <div id="hdr-ts" class="hdr-uptime" style="margin-right:4px">--:--:--</div>
    <div class="hdr-sep">│</div>
    <div id="hdr-uptime">UPTIME --:--:--</div>
    <div class="hdr-sep">│</div>
    <div id="hdr-cycle">CYCLE ---</div>
    <div class="pulse" id="hdr-pulse"></div>
  </div>
</div>

<!-- Warning stale banner -->
<div id="stale-banner" style="display:none; background:rgba(255,59,107,.12); border-bottom:1px solid rgba(255,59,107,.3); padding:8px; text-align:center; font-family:var(--mono); font-size:11px; color:var(--red); letter-spacing:1px; z-index:99; font-weight:700">
  ⚠️ ENGINE DISCONNECTED — SHOWING HISTORICAL CACHED DATA
</div>

<!-- OVERVIEW PAGE -->
<div id="overview-page">
  <!-- Main grid -->
  <div id="grid">

    <!-- LEFT: Currency Strengths + Burst + DER -->
    <div class="panel" id="panel-left">
      <div class="panel-title"><span class="title-dot"></span>CURRENCY MATRIX</div>

      <div class="panel-title" style="color:var(--cyan);margin-top:0;margin-bottom:8px;font-size:9px">WLS STRENGTHS</div>
      <div id="strengths-list"></div>

      <div class="divider"></div>
      <div class="panel-title" style="color:var(--amber);margin-bottom:8px;font-size:9px">BURST ACTIVITY</div>
      <div id="burst-list"></div>

      <div class="divider"></div>
      <div class="panel-title" style="color:var(--purple);margin-bottom:8px;font-size:9px">DIRECTIONAL EFFICIENCY</div>
      <div id="der-list"></div>
    </div>

    <!-- MIDDLE: Pipeline + Graph + Hypothesis -->
    <div class="panel" id="panel-pipeline">
      <div class="panel-title"><span class="title-dot" style="background:var(--purple)"></span>SIGNAL PIPELINE</div>

      <div class="funnel-wrap" id="funnel"></div>

      <div class="divider"></div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:center">
        <div>
          <div class="panel-title" style="font-size:9px;margin-bottom:6px;color:var(--cyan)">GRAPH HEALTH</div>
          <div class="gauge-wrap">
            <div class="gauge">
              <svg width="90" height="90" viewBox="0 0 90 90">
                <circle class="gauge-bg" cx="45" cy="45" r="36" stroke-width="7"/>
                <circle id="gauge-graph" class="gauge-fill" cx="45" cy="45" r="36" stroke-width="7"
                  stroke="var(--cyan)" stroke-dasharray="226.19" stroke-dashoffset="226.19"/>
              </svg>
              <div class="gauge-label">
                <div class="gauge-val" id="gauge-graph-val" style="color:var(--cyan)">--</div>
                <div class="gauge-sub">QUALITY</div>
              </div>
            </div>
          </div>
        </div>
        <div>
          <div class="panel-title" style="font-size:9px;margin-bottom:6px;color:var(--green)">TICK QUALITY</div>
          <div class="gauge-wrap">
            <div class="gauge">
              <svg width="90" height="90" viewBox="0 0 90 90">
                <circle class="gauge-bg" cx="45" cy="45" r="36" stroke-width="7"/>
                <circle id="gauge-tick" class="gauge-fill" cx="45" cy="45" r="36" stroke-width="7"
                  stroke="var(--green)" stroke-dasharray="226.19" stroke-dashoffset="226.19"/>
              </svg>
              <div class="gauge-label">
                <div class="gauge-val" id="gauge-tick-val" style="color:var(--green)">--</div>
                <div class="gauge-sub">TICK</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="divider"></div>
      <div class="panel-title" style="font-size:9px;margin-bottom:6px;color:var(--amber)">TOP HYPOTHESIS</div>
      <div id="hypothesis"></div>
    </div>

    <!-- RIGHT: Positions + Risk + Universe -->
    <div style="display:flex;flex-direction:column;gap:12px">

      <div class="panel">
        <div class="panel-title"><span class="title-dot" style="background:var(--green)"></span>OPEN POSITIONS</div>
        <div id="positions"></div>
        <div style="margin-top:8px;font-family:var(--mono);font-size:12px;display:flex;justify-content:space-between">
          <span style="color:var(--text3)">TOTAL P&amp;L</span>
          <span id="total-pnl" style="font-weight:700">$0.00</span>
        </div>
      </div>

      <div class="panel">
        <div class="panel-title"><span class="title-dot" style="background:var(--amber)"></span>RISK ENGINE</div>
        <div class="risk-row">
          <span class="risk-label">LOT SIZE</span>
          <span class="risk-val" id="risk-lot">--</span>
        </div>
        <div class="risk-row">
          <span class="risk-label">OPEN / MAX LOTS</span>
          <span class="risk-val" id="risk-lots">--</span>
        </div>
        <div class="lots-bar-wrap"><div class="lots-bar" id="lots-bar" style="width:0%"></div></div>
        <div class="risk-row">
          <span class="risk-label">PROFIT TARGET</span>
          <span class="risk-val" id="risk-target">--</span>
        </div>
        <div class="risk-row">
          <span class="risk-label">STOP LOSS</span>
          <span class="risk-val" id="risk-stop" style="color:var(--red)">--</span>
        </div>
        <div id="cooldown-badge"></div>
      </div>

      <div class="panel">
        <div class="panel-title"><span class="title-dot" style="background:var(--purple)"></span>UNIVERSE</div>
        <div class="uni-row">
          <span class="risk-label">SYMBOLS</span>
          <span class="uni-val" id="uni-val" style="color:var(--cyan)">--/--</span>
          <div class="uni-bar-wrap"><div class="uni-bar" id="uni-bar" style="width:0%"></div></div>
        </div>
        <div class="risk-row" style="margin-top:4px">
          <span class="risk-label">MISSING</span>
          <span class="risk-val" id="uni-missing" style="color:var(--amber)">0</span>
        </div>
        <div class="risk-row">
          <span class="risk-label">GRAPH CONF</span>
          <span class="risk-val" id="uni-conf">--</span>
        </div>
        <div class="risk-row">
          <span class="risk-label">SOLVE LATENCY</span>
          <span class="risk-val" id="uni-latency">-- ms</span>
        </div>
        <div class="risk-row">
          <span class="risk-label">MEMORY</span>
          <span class="risk-val" id="uni-mem">-- MB</span>
        </div>
      </div>
    </div>

  </div>
</div>

<!-- SWING ANALYTICS PAGE (hidden by default) -->
<div id="analytics-page" style="display:none; padding:12px 14px">

  <!-- Swing Stop overlay panel -->
  <div class="panel" id="panel-swing" style="margin-bottom:12px">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
      <span class="panel-title" style="margin-bottom:0"><span class="title-dot" style="background:var(--amber)"></span>SWING STOPS &amp; VOLATILITY (M5 Ranges)</span>
      <span id="swing-max-sl-badge" style="font-family:var(--mono);font-size:10px;color:var(--red);background:rgba(255,59,107,.1);padding:2px 8px;border-radius:4px;border:1px solid rgba(255,59,107,.2)">MAX SL: -$60.00</span>
      <span style="margin-left:auto;font-family:var(--mono);font-size:10px;color:var(--text3)">pips from M5 open | TP = entry ± remaining × 0.80</span>
    </div>
    <div id="swing-table-wrap" style="background:var(--bg2);border:1px solid var(--border);border-radius:var(--r2);overflow:hidden">
      <table id="swing-table" style="width:100%;border-collapse:collapse;font-family:var(--mono);font-size:10px">
        <thead>
          <tr style="background:var(--bg3);border-bottom:1px solid var(--border)">
            <th style="padding:6px 10px;text-align:left;color:var(--text3);letter-spacing:1px">SYM</th>
            <th style="padding:6px 10px;text-align:center;color:var(--text3);letter-spacing:1px">SWING %</th>
            <th style="padding:6px 10px;text-align:center;color:var(--text3);letter-spacing:1px">RANGE</th>
            <th style="padding:6px 10px;text-align:right;color:var(--text3);letter-spacing:1px">▼DN</th>
            <th style="padding:6px 10px;text-align:right;color:var(--text3);letter-spacing:1px">▲UP</th>
            <th style="padding:6px 10px;text-align:center;color:var(--text3);letter-spacing:1px">BAR POS</th>
            <th style="padding:6px 10px;text-align:right;color:var(--text3);letter-spacing:1px">BUY SL</th>
            <th style="padding:6px 10px;text-align:right;color:var(--text3);letter-spacing:1px">BUY SL ($)</th>
            <th style="padding:6px 10px;text-align:right;color:var(--text3);letter-spacing:1px">BUY TP</th>
            <th style="padding:6px 10px;text-align:right;color:var(--text3);letter-spacing:1px">BUY TP ($)</th>
            <th style="padding:6px 10px;text-align:right;color:var(--text3);letter-spacing:1px">SELL SL</th>
            <th style="padding:6px 10px;text-align:right;color:var(--text3);letter-spacing:1px">SELL SL ($)</th>
            <th style="padding:6px 10px;text-align:right;color:var(--text3);letter-spacing:1px">SELL TP</th>
            <th style="padding:6px 10px;text-align:right;color:var(--text3);letter-spacing:1px">SELL TP ($)</th>
            <th style="padding:6px 10px;text-align:center;color:var(--text3);letter-spacing:1px">HISTORY</th>
          </tr>
        </thead>
        <tbody id="swing-tbody"></tbody>
      </table>
    </div>
  </div>
</div>



<!-- BOTTOM STATUS BAR -->
  <div id="statusbar">
    <div class="sb-chip chip-ok" id="chip-mt5"><span class="sb-label">MT5</span>&nbsp;<span class="sb-val">--</span></div>
    <div class="sb-sep">│</div>
    <div class="sb-chip chip-neu" id="chip-tick"><span class="sb-label">TICK</span>&nbsp;<span class="sb-val">--</span></div>
    <div class="sb-sep">│</div>
    <div class="sb-chip chip-neu" id="chip-graph"><span class="sb-label">GRAPH</span>&nbsp;<span class="sb-val">--</span></div>
    <div class="sb-sep">│</div>
    <div class="sb-chip chip-neu" id="chip-snap"><span class="sb-label">SNAP</span>&nbsp;<span class="sb-val">--</span></div>
    <div class="sb-sep">│</div>
    <div class="sb-chip chip-neu" id="chip-solve"><span class="sb-label">SOLVE</span>&nbsp;<span class="sb-val">-- ms</span></div>
    <div class="sb-sep">│</div>
    <div class="sb-chip chip-neu" id="chip-mem"><span class="sb-label">MEM</span>&nbsp;<span class="sb-val">-- MB</span></div>
    <div class="sb-sep">│</div>
    <div class="sb-chip chip-neu" id="chip-mode"><span class="sb-label">MODE</span>&nbsp;<span class="sb-val">--</span></div>
    <div class="sb-sep" style="margin-left:auto">│</div>
    <div style="font-family:var(--mono);font-size:10px;color:var(--text3)">LAST UPDATE: <span id="last-update">--</span></div>
  </div>

</div>

<script>
const CCY_ORDER = ['EUR','USD','GBP','JPY','CHF','AUD','CAD','NZD'];

function fmt(v, dec=5){ return v >= 0 ? `+${v.toFixed(dec)}` : v.toFixed(dec); }
function fmtPnl(v){ return (v >= 0 ? '+' : '') + '$' + Math.abs(v).toFixed(2); }

function clamp(v,mn,mx){ return Math.max(mn, Math.min(mx, v)); }

function gaugeSet(id, valId, value, colorGood='var(--cyan)', colorBad='var(--red)'){
  const r = 36, circ = 2 * Math.PI * r;
  const pct = clamp(value, 0, 1);
  const offset = circ * (1 - pct);
  const el = document.getElementById(id);
  const ve = document.getElementById(valId);
  if(el){ el.style.strokeDashoffset = offset; el.style.stroke = pct > 0.5 ? colorGood : pct > 0.3 ? 'var(--amber)' : colorBad; }
  if(ve){ ve.textContent = (value*100).toFixed(0) + '%'; ve.style.color = pct > 0.5 ? colorGood : pct > 0.3 ? 'var(--amber)' : colorBad; }
}

const CCY_COLOR_MAP = {
  'EUR': 'var(--green)',
  'USD': 'var(--green)',
  'GBP': 'var(--text)',
  'JPY': 'var(--text)',
  'CHF': 'var(--text)',
  'AUD': 'var(--text)',
  'CAD': 'var(--red)',
  'NZD': 'var(--red)'
};

function chipClass(val, threshOk=0.5, threshWarn=0.3){
  if(val >= threshOk) return 'chip-ok';
  if(val >= threshWarn) return 'chip-warn';
  return 'chip-err';
}

function renderCcySection(containerId, data, maxAbsVal, decimalPlaces=5, checkPers=false, strengthsMap=null){
  if(!data) return;
  const el = document.getElementById(containerId);
  if(!el) return;
  const entries = CCY_ORDER.map(c => [c, data[c] || 0]);
  const sorted = [...entries].sort((a,b) => Math.abs(b[1])-Math.abs(a[1]));
  const maxA = Math.max(...entries.map(e => Math.abs(e[1])), 1e-12);
  let html = '';
  sorted.forEach(([ccy, val], i) => {
    const defaultColor = CCY_COLOR_MAP[ccy] || 'var(--text)';
    const rankCls = i === 0 ? 'rank-top' : i === sorted.length-1 ? 'rank-bot' : '';
    const dir = val > 1e-8 ? 'pos' : val < -1e-8 ? 'neg' : 'neu';
    const pct = clamp(Math.abs(val)/maxA, 0, 1)*100;
    
    let arrow = '○';
    let streak = 0;
    let extStr = '─';
    if(checkPers && strengthsMap && strengthsMap[ccy]){
      const sinfo = strengthsMap[ccy];
      const sdir = sinfo.dir || 0;
      streak = sinfo.streak || 0;
      arrow = sdir > 0 ? '▲' : sdir < 0 ? '▼' : '○';
    } else {
      arrow = val > 1e-8 ? '▲' : val < -1e-8 ? '▼' : '○';
    }

    html += `<div class="ccy-row ${rankCls}">
      <span class="ccy-label" style="color:${rankCls===''?defaultColor:''}">${ccy}</span>
      <div class="ccy-bar-wrap"><div class="ccy-bar ${dir}" style="width:${pct.toFixed(1)}%"></div></div>
      <span class="ccy-val" style="font-size:10px;color:${rankCls===''?defaultColor:''}">${fmt(val, decimalPlaces)}</span>
      <span class="ccy-meta">${arrow}${streak > 0 ? streak : ''}</span>
    </div>`;
  });
  el.innerHTML = html;
}

function renderPipeline(metrics){
  if(!metrics) return;
  const el = document.getElementById('funnel');
  if(!el) return;
  const gen  = metrics.generated || 0;
  const burst= metrics.burst_hyp || 0;
  const bar  = metrics.bar_aligned || 0;
  const rnk  = metrics.ranked || 0;
  const sel  = metrics.selected || 0;
  const risk = metrics.risk_approved || 0;
  const exe  = metrics.executed || 0;
  const stages = [
    { label:'GENERATED', val:gen,   max:gen||1,  cls:'f-gen',      color:'var(--purple)' },
    { label:'BURST FILT',val:burst, max:gen||1,  cls:'f-burst',    color:'var(--cyan)'   },
    { label:'BAR ALIGN', val:bar,   max:gen||1,  cls:'f-bar',      color:'var(--cyan2)'  },
    { label:'RANKED',    val:rnk,   max:gen||1,  cls:'f-ranked',   color:'var(--amber)'  },
    { label:'SELECTED',  val:sel,   max:gen||1,  cls:'f-selected', color:'var(--green2)' },
    { label:'RISK OK',   val:risk,  max:gen||1,  cls:'f-risk',     color:'var(--green)'  },
    { label:'EXECUTED',  val:exe,   max:gen||1,  cls:'f-exe',      color:'var(--green)'  },
  ];
  let html = '';
  stages.forEach((s, i) => {
    const pct = gen > 0 ? clamp(s.val/(gen), 0, 1)*100 : (s.val > 0 ? 100 : 0);
    const minPct = s.val > 0 ? Math.max(pct, 8) : 0;
    const isLast = i === stages.length-1;
    html += `<div class="funnel-stage">
      <span class="funnel-label">${s.label}</span>
      <div class="funnel-bar-wrap">
        <div class="funnel-bar ${s.cls}" style="width:${minPct.toFixed(1)}%;min-width:${s.val>0?32:0}px">
          ${s.val}
        </div>
      </div>
      ${!isLast ? `<span class="funnel-arrow">›</span>` : ''}
    </div>`;
  });
  el.innerHTML = html;
}

function renderPositions(posCount, pnl){
  const el = document.getElementById('positions');
  if(!el) return;
  if(posCount === 0){
    el.innerHTML = '<div class="no-pos">NO OPEN POSITIONS</div>';
  } else {
    let html = '';
    for(let i = 0; i < posCount; i++){
      html += `<div class="pos-card"><div class="pos-header">
        <span class="pos-sym">POSITION ${i+1}</span>
        <span class="pos-dir buy">OPEN</span>
      </div></div>`;
    }
    el.innerHTML = html;
  }
  const totalEl = document.getElementById('total-pnl');
  if(totalEl){ 
    totalEl.textContent = fmtPnl(pnl || 0);
    totalEl.style.color = (pnl||0) >= 0 ? 'var(--green)' : 'var(--red)';
  }
}

function renderHypothesis(top){
  const el = document.getElementById('hypothesis');
  if(!el) return;
  if(!top || !top.top_symbol){
    el.innerHTML = `<div style="color:var(--text3);font-family:var(--mono);font-size:11px;padding:12px;text-align:center;border:1px dashed var(--border);border-radius:var(--r)">NO SIGNAL — BELOW MIN CONFIDENCE</div>`;
    return;
  }
  const dir = top.top_hypothesis_dir;
  const dirCls = dir > 0 ? 'buy' : 'sell';
  const dirLbl = dir > 0 ? 'BUY' : 'SELL';
  const conf = (top.top_conf || 0);
  const drs  = (top.top_drs  || 0);
  el.innerHTML = `<div class="hyp-card fade-in">
    <div style="display:flex;align-items:baseline">
      <span class="hyp-sym">${top.top_symbol}</span>
      <span class="hyp-dir ${dirCls}">${dirLbl}</span>
    </div>
    <div class="hyp-row">
      <div class="hyp-metric">
        <span class="hyp-mlabel">CONFIDENCE</span>
        <span class="hyp-mval" style="color:${conf>0.6?'var(--green)':conf>0.4?'var(--amber)':'var(--red)'}">${(conf*100).toFixed(1)}%</span>
      </div>
      <div class="hyp-metric">
        <span class="hyp-mlabel">DRS SCORE</span>
        <span class="hyp-mval" style="color:var(--cyan)">${drs.toFixed(3)}</span>
      </div>
    </div>
  </div>`;
}

function renderStatusBar(d, active){
  function setChip(id, label, val, cls){
    const el = document.getElementById(id);
    if(!el) return;
    el.className = `sb-chip ${cls}`;
    el.innerHTML = `<span class="sb-label">${label}</span>&nbsp;<span class="sb-val">${val}</span>`;
  }
  
  if (!active) {
    setChip('chip-mt5','MT5','OFFLINE', 'chip-err');
    setChip('chip-tick','TICK','OFFLINE', 'chip-err');
    setChip('chip-graph','GRAPH','OFFLINE', 'chip-err');
    setChip('chip-snap','SNAP','STALE', 'chip-warn');
    setChip('chip-solve','SOLVE','-- ms', 'chip-neu');
    setChip('chip-mem','MEM','-- MB', 'chip-neu');
    setChip('chip-mode','MODE','OFFLINE', 'chip-err');
  } else {
    const mt5ok = d.mt5_ok;
    setChip('chip-mt5','MT5',mt5ok?'OK':'FAIL', mt5ok?'chip-ok':'chip-err');
    setChip('chip-tick','TICK',(d.tick_quality||0).toFixed(2), chipClass(d.tick_quality||0));
    setChip('chip-graph','GRAPH',(d.graph_quality||0).toFixed(2), chipClass(d.graph_quality||0));
    setChip('chip-snap','SNAP','OK','chip-ok');
    setChip('chip-solve','SOLVE',`${(d.solve_latency_ms||0).toFixed(0)} ms`, 'chip-neu');
    setChip('chip-mem','MEM',`${(d.memory_mb||0).toFixed(0)} MB`,'chip-neu');
    const mode = d.mode || '--';
    setChip('chip-mode','MODE', mode, mode==='LIVE'?'chip-ok':'chip-warn');
  }
}

function update(d){
  // Hide no-data overlay
  document.getElementById('nodata').classList.add('hidden');

  const active = d.engine_active;

  // Toggle warning banner visibility
  const staleEl = document.getElementById('stale-banner');
  if (staleEl) {
    staleEl.style.display = active ? 'none' : 'block';
  }

  // Header Mode and Status updating
  document.getElementById('hdr-ts').textContent = d.ts || '--';
  const uptime = d.uptime || 0;
  const uh = String(Math.floor(uptime/3600)).padStart(2,'0');
  const um = String(Math.floor((uptime%3600)/60)).padStart(2,'0');
  const us = String(Math.floor(uptime%60)).padStart(2,'0');
  document.getElementById('hdr-uptime').textContent = `UPTIME ${uh}:${um}:${us}`;
  document.getElementById('hdr-cycle').textContent = `CYCLE ${d.trade_count||0}`;

  const pulseEl = document.getElementById('hdr-pulse');
  const hstate = d.health_state || 'OK';
  const hel = document.getElementById('hdr-health');
  const mel = document.getElementById('hdr-mode');

  if (!active) {
    mel.textContent = 'OFFLINE';
    mel.className = 'hdr-pill pill-err';
    if (pulseEl) {
      pulseEl.style.background = 'var(--red)';
      pulseEl.style.boxShadow = '0 0 8px var(--red)';
      pulseEl.style.animation = 'none';
    }
    hel.className = 'hdr-pill pill-err';
    hel.textContent = '● DISCONNECTED';
  } else {
    mel.textContent = d.mode || 'LIVE';
    mel.className = 'hdr-pill pill-mode';
    if (pulseEl) {
      pulseEl.style.background = 'var(--green)';
      pulseEl.style.boxShadow = '0 0 8px var(--green)';
      pulseEl.style.animation = 'pulse 2s infinite';
    }
    if(hstate==='OK'){ hel.className='hdr-pill pill-ok'; hel.textContent='● OK'; }
    else if(hstate==='DEGRADED'){ hel.className='hdr-pill pill-warn'; hel.textContent='▲ DEGRADED'; }
    else { hel.className='hdr-pill pill-err'; hel.textContent='✕ ERROR'; }
  }

  // Currency sections
  renderCcySection('strengths-list', d.currency_strengths, null, 5, true, d.strength_persistence);
  renderCcySection('burst-list', d.currency_bursts, null, 3, true, d.burst_persistence);
  renderCcySection('der-list', d.observability, null, 2, false);

  // Gauges
  gaugeSet('gauge-graph','gauge-graph-val', d.graph_quality||0, 'var(--cyan)', 'var(--red)');
  gaugeSet('gauge-tick','gauge-tick-val', d.tick_quality||0, 'var(--green)', 'var(--red)');

  // Pipeline
  renderPipeline(d.pipeline);

  // Positions
  renderPositions(d.positions||0, d.pnl||0);

  // Hypothesis
  renderHypothesis(d);

  // Risk
  document.getElementById('risk-lot').textContent = (d.lot_size||0.70).toFixed(2);
  const maxLots = d.max_total_lots || 2.10;
  const openLots = d.open_lots || 0;
  document.getElementById('risk-lots').textContent = `${openLots.toFixed(2)} / ${maxLots.toFixed(2)}`;
  const lotsPct = maxLots > 0 ? clamp(openLots/maxLots,0,1)*100 : 0;
  document.getElementById('lots-bar').style.width = lotsPct.toFixed(1)+'%';
  document.getElementById('risk-target').textContent = `$${(d.profit_target||50).toFixed(0)}`;
  document.getElementById('risk-stop').textContent = `-$${Math.abs(d.stop_loss_amount||60).toFixed(0)}`;
  const swingMaxSl = document.getElementById('swing-max-sl-badge');
  if (swingMaxSl) {
    swingMaxSl.textContent = `MAX SL: -$${Math.abs(d.stop_loss_amount||60).toFixed(0)}`;
  }
  const cdEl = document.getElementById('cooldown-badge');
  if(d.cooldown_active){
    cdEl.innerHTML = `<div class="cooldown-badge">⚡ COOLDOWN ${d.cooldown_remaining||0}s</div>`;
  } else { cdEl.innerHTML=''; }

  // Universe
  const avail = d.universe_available || 0;
  const cfg   = d.universe_configured || 0;
  document.getElementById('uni-val').textContent = `${avail}/${cfg}`;
  document.getElementById('uni-val').style.color = avail===cfg?'var(--green)':'var(--amber)';
  document.getElementById('uni-bar').style.width = (cfg>0?clamp(avail/cfg,0,1)*100:0).toFixed(1)+'%';
  document.getElementById('uni-missing').textContent = d.missing_symbols||0;
  document.getElementById('uni-conf').textContent = d.health_conf || '--';
  document.getElementById('uni-latency').textContent = `${(d.solve_latency_ms||0).toFixed(0)} ms`;
  document.getElementById('uni-mem').textContent = `${(d.memory_mb||0).toFixed(1)} MB`;

  // Status bar
  renderStatusBar(d, active);

  // Last update
  document.getElementById('last-update').textContent = new Date().toLocaleTimeString();



  // Swing overlay
  renderSwing(d);
}

// ── Swing Overlay ────────────────────────────────────────────────

function renderSwing(data){
  const tbody = document.getElementById('swing-tbody');
  if(!tbody) return;
  const overlay = data.swing_overlay || {};
  const syms = Object.keys(overlay).sort();
  const activeSymbols = data.active_symbols || [];
  let html = '';
  syms.forEach(sym => {
    const sw = overlay[sym];
    const dn = sw.avg_down || 0;
    const up = sw.avg_up || 0;
    const fp = sw.forming_pips || 0;
    const total = Math.max(up - dn, 1);
    const posPct = ((fp - dn) / total * 100);
    const pctClamped = Math.max(2, Math.min(98, posPct));
    const barCls = fp > 0.5 ? 'pos' : fp < -0.5 ? 'neg' : 'mid';
    const isActive = activeSymbols.includes(sym);
    
    // Swing reach column
    let reachHtml = '<span style="color:var(--text3)">--</span>';
    if (sw.swing_reach) {
      const r = sw.swing_reach;
      const pct = Math.max(0, r.pct);
      const fillW = Math.min(pct, 100);
      const cls = pct >= 120 ? 'reach-over' : pct >= 90 ? 'reach-high' : pct >= 50 ? 'reach-mid' : 'reach-low';
      const pctColor = pct >= 120 ? 'var(--red)' : pct >= 90 ? 'var(--green)' : pct >= 50 ? 'var(--amber)' : 'var(--cyan)';
      const dirArrow = r.direction === 'BUY' ? '▲' : '▼';
      const pnlColor = r.pnl >= 0 ? 'var(--green)' : 'var(--red)';
      const pnlStr = r.pnl >= 0 ? '+$'+r.pnl.toFixed(0) : '-$'+Math.abs(r.pnl).toFixed(0);
      reachHtml = `<div class="swing-reach-bar-wrap">
        <span style="font-size:9px;color:${pctColor};font-weight:700;min-width:32px;text-align:right">${pct.toFixed(0)}%</span>
        <div class="swing-reach-bar">
          <div class="swing-reach-fill ${cls}" style="width:${fillW}%"></div>
        </div>
        <span style="font-size:8px;color:var(--text3);min-width:28px">${r.moved_pips}p/${r.expected_pips}p</span>
      </div>
      <div style="font-size:8px;color:var(--text2);text-align:center;margin-top:2px">${dirArrow} <span style="color:${pnlColor};font-weight:bold">${pnlStr}</span></div>`;
    }

    let histHtml = '--';
    if (sw.history) {
      const h = sw.history;
      if (h.status === 'RUNNING') {
        const peak = h.peak_pnl || 0;
        const color = peak >= 0 ? 'var(--green)' : 'var(--red)';
        const peakStr = peak >= 0 ? '+$' + peak.toFixed(0) : '-$' + Math.abs(peak).toFixed(0);
        histHtml = `RUN: ${h.direction} @ ${h.entry_price} (<span style="color:${color};font-weight:bold">${peakStr}</span> peak)`;
      } else if (h.status === 'CLOSED') {
        const pnl = h.pnl || 0;
        const color = pnl >= 0 ? 'var(--green)' : 'var(--red)';
        const pnlStr = pnl >= 0 ? '+$' + pnl.toFixed(0) : '-$' + Math.abs(pnl).toFixed(0);
        histHtml = `HIT ${h.reason} @ ${h.exit_price} (<span style="color:${color};font-weight:bold">${pnlStr}</span>)`;
      }
    }

    html += `<tr class="${isActive ? 'active-trade-row' : ''}">
      <td>${sym}</td>
      <td style="padding:4px 6px;min-width:130px">${reachHtml}</td>
      <td>
        <div class="swing-bar-wrap">
          <span style="color:var(--red);font-size:9px;width:28px;text-align:right">${dn.toFixed(0)}</span>
          <div class="swing-bar" style="flex:1;min-width:60px">
            <div class="swing-fill ${barCls}" style="left:0;width:100%;opacity:0.15;background:var(--amber)"></div>
            <div class="swing-dot" style="left:${pctClamped}%"></div>
          </div>
          <span style="color:var(--green);font-size:9px;width:28px">${up.toFixed(0)}</span>
        </div>
      </td>
      <td style="text-align:right;color:var(--red)">${dn.toFixed(1)}</td>
      <td style="text-align:right;color:var(--green)">+${up.toFixed(1)}</td>
      <td style="text-align:center;color:${fp > 1 ? 'var(--green)' : fp < -1 ? 'var(--red)' : 'var(--text2)'}">${fp > 0 ? '+' : ''}${fp.toFixed(1)}p</td>
      <td style="text-align:right;color:var(--red)">${sw.buy_sl || '--'}</td>
      <td style="text-align:right;color:var(--red);font-weight:bold">${sw.buy_sl_usd !== undefined ? '-$' + sw.buy_sl_usd.toFixed(0) : '--'}</td>
      <td style="text-align:right;color:var(--green)">${sw.buy_tp || '--'}</td>
      <td style="text-align:right;color:var(--green);font-weight:bold">${sw.buy_tp_usd !== undefined ? '+$' + sw.buy_tp_usd.toFixed(0) : '--'}</td>
      <td style="text-align:right;color:var(--red)">${sw.sell_sl || '--'}</td>
      <td style="text-align:right;color:var(--red);font-weight:bold">${sw.sell_sl_usd !== undefined ? '-$' + sw.sell_sl_usd.toFixed(0) : '--'}</td>
      <td style="text-align:right;color:var(--green)">${sw.sell_tp || '--'}</td>
      <td style="text-align:right;color:var(--green);font-weight:bold">${sw.sell_tp_usd !== undefined ? '+$' + sw.sell_tp_usd.toFixed(0) : '--'}</td>
      <td style="text-align:center;color:var(--text2);font-size:9px">${histHtml}</td>
    </tr>`;
  });
  tbody.innerHTML = html;
}







// ── Navigation toggle ────────────────────────────────────────────
document.addEventListener('click', function(e){
  const navOverview = e.target.closest('#nav-overview');
  const navSwing = e.target.closest('#nav-swing');
  
  if(navOverview) {
    document.getElementById('overview-page').style.display = '';
    document.getElementById('analytics-page').style.display = 'none';
    document.getElementById('nav-overview').classList.add('active');
    document.getElementById('nav-swing').classList.remove('active');
  }
  
  if(navSwing) {
    document.getElementById('overview-page').style.display = 'none';
    document.getElementById('analytics-page').style.display = 'block';
    document.getElementById('nav-overview').classList.remove('active');
    document.getElementById('nav-swing').classList.add('active');
  }
});

// ── SSE connection ────────────────────────────────────────────────
function connect(){
  const es = new EventSource('/events');
  es.onmessage = (e) => {
    try{ update(JSON.parse(e.data)); } catch(ex){ console.warn('parse error', ex); }
  };
  es.onerror = () => {
    es.close();
    setTimeout(connect, 3000);
  };
}
connect();
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
#  HTTP Handler
# ─────────────────────────────────────────────────────────────────────────────
class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress access log spam

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_html()
        elif self.path == "/events":
            self._serve_sse()
        elif self.path == "/data":
            self._serve_json()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_html(self):
        body = _HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_json(self):
        with _state_lock:
            body = json.dumps(_state).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _serve_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        # Send current state immediately if available
        with _state_lock:
            initial = _state.copy()
        if initial:
            try:
                self.wfile.write(f"data: {json.dumps(initial)}\n\n".encode())
                self.wfile.flush()
            except Exception:
                return

        # Register for push updates
        with _sse_lock:
            _sse_clients.append(self.wfile)

        # Keep alive: send comment every 15s (client reconnects on error)
        try:
            while True:
                time.sleep(15)
                self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except Exception:
            pass
        finally:
            with _sse_lock:
                if self.wfile in _sse_clients:
                    _sse_clients.remove(self.wfile)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Validate log file path
    if not LATEST_FILE.parent.exists():
        print(f"[WARN] Log directory not found: {LATEST_FILE.parent}")
        print(f"       Dashboard will wait for the engine to create it.")

    # Start poller thread — daemon so it dies with the process
    t = threading.Thread(target=_poller_loop, daemon=True)
    t.start()

    server = HTTPServer(("0.0.0.0", PORT), _Handler)
    print(f"")
    print(f"  ⬡  PROXIMA CDE — Game Dashboard")
    print(f"  ──────────────────────────────────────────")
    print(f"  Open: http://localhost:{PORT}")
    print(f"  Data: {LATEST_FILE}")
    print(f"  Poll: every {POLL_SEC:.1f}s  |  Zero impact on trading engine")
    print(f"  ──────────────────────────────────────────")
    print(f"  Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")


if __name__ == "__main__":
    main()
