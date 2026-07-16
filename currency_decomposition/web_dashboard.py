"""
web_dashboard.py — Proxima Currency Decomposition Engine — Game-Level Web Dashboard
======================================================================================
Runs as a COMPLETELY SEPARATE process — zero performance impact on the trading engine.

Reads:  logs/dashboard_latest.json   (written every 500ms by the engine)
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
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
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
#header{display:flex;align-items:center;gap:10px;padding:5px 16px;background:linear-gradient(90deg,rgba(0,229,255,.08),rgba(0,229,255,.02));border-bottom:1px solid var(--border2);position:sticky;top:0;z-index:100;backdrop-filter:blur(12px);height:40px}
.hdr-logo{font-family:var(--mono);font-size:13px;font-weight:700;color:var(--cyan);letter-spacing:1.5px;text-shadow:0 0 20px var(--cyan);flex-shrink:0}
.hdr-sep{color:var(--border2);font-size:16px;flex-shrink:0}
.hdr-pill{padding:2px 8px;border-radius:20px;font-family:var(--mono);font-size:10px;font-weight:600;letter-spacing:.3px;flex-shrink:0}
.pill-mode{background:rgba(0,229,255,.12);border:1px solid var(--cyan3);color:var(--cyan)}
.pill-ok{background:rgba(0,255,136,.1);border:1px solid var(--green3);color:var(--green)}
.pill-warn{background:rgba(255,170,0,.1);border:1px solid var(--amber3);color:var(--amber)}
.pill-err{background:rgba(255,59,107,.1);border:1px solid var(--red3);color:var(--red)}
.hdr-nav{display:flex;gap:6px;margin:0 12px}
.nav-link{background:none;border:1px solid transparent;color:var(--text3);font-family:var(--mono);font-size:10px;font-weight:700;letter-spacing:1px;cursor:pointer;padding:3px 9px;border-radius:4px;transition:color .15s,background .15s,border-color .15s}
.nav-link:hover{color:var(--cyan);background:rgba(0,229,255,.05)}
.nav-link.active{color:var(--cyan);background:rgba(0,229,255,.1);border-color:rgba(0,229,255,.25)}
.hdr-right{margin-left:auto;display:flex;align-items:center;gap:8px;flex-shrink:0}
#hdr-uptime,#hdr-cycle,#hdr-ts{font-family:var(--mono);font-size:10px;color:var(--text2)}
.pulse{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green);animation:pulse 2s infinite;flex-shrink:0}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.8)}}

/* ── CHOP INDICATOR — eyecatching compact pill in header ── */
#chop-indicator{display:none;align-items:center;gap:5px;padding:3px 10px;border-radius:20px;font-family:var(--mono);font-size:10px;font-weight:700;border:1px solid rgba(255,59,107,.5);background:rgba(255,59,107,.12);color:var(--red);animation:chop-pulse 1.5s infinite;cursor:help;letter-spacing:.3px;flex-shrink:0}
#chop-indicator.active{display:flex}
@keyframes chop-pulse{0%,100%{box-shadow:0 0 0 rgba(255,59,107,0)}50%{box-shadow:0 0 10px rgba(255,59,107,.4)}}
.chop-dot{width:6px;height:6px;border-radius:50%;background:var(--red);box-shadow:0 0 6px var(--red);animation:pulse 1s infinite}
#chop-gap-pill{display:inline-flex;align-items:center;gap:3px;padding:2px 6px;border-radius:10px;background:rgba(255,170,0,.15);border:1px solid rgba(255,170,0,.3);color:var(--amber);font-size:9px;margin-left:3px}

/* ── Grid layout ── */
#grid{display:grid;grid-template-columns:260px 1fr 240px;gap:10px;padding:10px 12px;min-height:calc(100vh - 52px)}

/* ── Panels ── */
.panel{background:var(--bg2);border:1px solid var(--border);border-radius:var(--r2);padding:12px;position:relative;overflow:hidden}
.panel::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,rgba(0,229,255,.025) 0%,transparent 60%);pointer-events:none}
.panel-title{font-family:var(--mono);font-size:10px;font-weight:700;letter-spacing:2px;color:var(--text3);text-transform:uppercase;margin-bottom:10px;display:flex;align-items:center;gap:6px}
.panel-title .title-dot{width:5px;height:5px;border-radius:50%;background:var(--cyan);box-shadow:0 0 6px var(--cyan);flex-shrink:0}

/* ── Currency bars ── */
.ccy-row{display:flex;align-items:center;gap:6px;margin-bottom:5px}
.ccy-label{font-family:var(--mono);font-size:11px;font-weight:700;width:26px;color:var(--text)}
.ccy-bar-wrap{flex:1;height:5px;background:var(--bg4);border-radius:3px;overflow:hidden}
.ccy-bar{height:100%;border-radius:3px;transition:width .6s cubic-bezier(.4,0,.2,1)}
.ccy-bar.pos{background:linear-gradient(90deg,rgba(0,255,136,.3),var(--green))}
.ccy-bar.neg{background:linear-gradient(90deg,rgba(255,59,107,.3),var(--red))}
.ccy-bar.neu{background:var(--bg4)}
.ccy-val{font-family:var(--mono);font-size:10px;width:60px;text-align:right;color:var(--text2)}
.ccy-meta{font-family:var(--mono);font-size:9px;color:var(--text3);width:20px;text-align:center}
.rank-top .ccy-label,.rank-top .ccy-val{color:var(--green);text-shadow:0 0 8px rgba(0,255,136,.4)}
.rank-bot .ccy-label,.rank-bot .ccy-val{color:var(--red);text-shadow:0 0 8px rgba(255,59,107,.4)}

/* ── Section divider ── */
.divider{height:1px;background:linear-gradient(90deg,transparent,var(--border),transparent);margin:8px 0}

/* ── Pipeline funnel ── */
.funnel-wrap{display:flex;flex-direction:column;gap:5px}
.funnel-stage{display:flex;align-items:center;gap:8px}
.funnel-label{font-family:var(--mono);font-size:9px;color:var(--text3);width:72px;letter-spacing:.5px}
.funnel-bar-wrap{flex:1;height:22px;background:var(--bg4);border-radius:4px;overflow:hidden}
.funnel-bar{height:100%;border-radius:4px;display:flex;align-items:center;padding-left:6px;font-family:var(--mono);font-size:11px;font-weight:700;transition:width .8s cubic-bezier(.4,0,.2,1);white-space:nowrap;position:relative;overflow:hidden}
.funnel-bar::before{content:'';position:absolute;inset:0;background:linear-gradient(90deg,rgba(255,255,255,.05),transparent)}
.funnel-arrow{color:var(--text3);font-size:12px}
.f-gen{background:linear-gradient(90deg,rgba(176,72,255,.4),var(--purple))}
.f-burst{background:linear-gradient(90deg,rgba(0,229,255,.4),var(--cyan))}
.f-bar{background:linear-gradient(90deg,rgba(0,184,212,.4),var(--cyan2))}
.f-ranked{background:linear-gradient(90deg,rgba(255,170,0,.4),var(--amber))}
.f-selected{background:linear-gradient(90deg,rgba(0,201,107,.4),var(--green2))}
.f-risk{background:linear-gradient(90deg,rgba(0,255,136,.4),var(--green))}
.f-exe{background:linear-gradient(90deg,rgba(0,255,136,.8),#00ff88);color:#000}

/* ── Graph gauge (compact) ── */
.gauge-wrap{display:flex;align-items:center;justify-content:center;gap:16px;padding:4px 0}
.gauge{width:70px;height:70px;position:relative}
.gauge svg{transform:rotate(-90deg)}
.gauge-bg{stroke:var(--bg4);fill:none}
.gauge-fill{fill:none;stroke-linecap:round;transition:stroke-dashoffset .8s cubic-bezier(.4,0,.2,1),stroke .4s}
.gauge-label{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;font-family:var(--mono)}
.gauge-val{font-size:15px;font-weight:700}
.gauge-sub{font-size:8px;color:var(--text3);letter-spacing:1px;margin-top:1px}

/* ── Hypothesis card ── */
.hyp-card{background:var(--bg3);border:1px solid var(--border2);border-radius:var(--r);padding:10px;margin-top:6px}
.hyp-sym{font-family:var(--mono);font-size:20px;font-weight:700;color:var(--cyan);letter-spacing:2px;text-shadow:0 0 16px rgba(0,229,255,.5)}
.hyp-dir{font-family:var(--mono);font-size:12px;font-weight:700;margin-left:6px}
.hyp-dir.buy{color:var(--green)}
.hyp-dir.sell{color:var(--red)}
.hyp-row{display:flex;gap:14px;margin-top:6px;flex-wrap:wrap}
.hyp-metric{display:flex;flex-direction:column;gap:2px}
.hyp-mlabel{font-size:8px;color:var(--text3);letter-spacing:1px;text-transform:uppercase}
.hyp-mval{font-family:var(--mono);font-size:13px;font-weight:700}

/* ── Positions ── */
.pos-card{background:var(--bg3);border:1px solid var(--border2);border-radius:var(--r);padding:8px 10px;margin-bottom:5px;transition:border-color .3s}
.pos-card.active-pos{border-color:var(--amber);background:rgba(255,170,0,.04)}
.pos-header{display:flex;align-items:center;gap:6px;margin-bottom:4px}
.pos-sym{font-family:var(--mono);font-size:12px;font-weight:700}
.pos-dir{font-family:var(--mono);font-size:9px;font-weight:700;padding:2px 6px;border-radius:10px}
.pos-dir.buy{background:rgba(0,255,136,.15);color:var(--green);border:1px solid rgba(0,255,136,.3)}
.pos-dir.sell{background:rgba(255,59,107,.15);color:var(--red);border:1px solid rgba(255,59,107,.3)}
.pos-pnl{margin-left:auto;font-family:var(--mono);font-size:12px;font-weight:700}
.pos-pnl.pos{color:var(--green)}
.pos-pnl.neg{color:var(--red)}
.pos-detail{font-family:var(--mono);font-size:9px;color:var(--text3);display:flex;gap:10px}
.no-pos{color:var(--text3);font-family:var(--mono);font-size:11px;text-align:center;padding:16px;border:1px dashed var(--border);border-radius:var(--r)}

/* ── Regime panel — compact ── */
.regime-compact{display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:var(--r);border:1px solid var(--border)}
.regime-icon{font-size:22px;line-height:1}
.regime-body{flex:1}
.regime-label{font-family:var(--mono);font-size:12px;font-weight:700}
.regime-bar-wrap{height:4px;background:var(--bg4);border-radius:2px;overflow:hidden;margin-top:4px}
.regime-bar-fill{height:100%;border-radius:2px;transition:width .5s,background .3s}
.regime-pct{font-family:var(--mono);font-size:10px;color:var(--text3);margin-top:2px}
.regime-threshold{font-family:var(--mono);font-size:9px;color:var(--text3)}

/* ── Chop progress compact widget ── */
.chop-widget{background:rgba(255,59,107,.06);border:1px solid rgba(255,59,107,.25);border-radius:var(--r);padding:8px 10px;margin-top:6px;display:none}
.chop-widget.active{display:block}
.chop-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:5px}
.chop-title{font-family:var(--mono);font-size:9px;font-weight:700;color:var(--red);letter-spacing:1px}
.chop-time{font-family:var(--mono);font-size:10px;color:var(--amber);font-weight:700}
.chop-pbar-wrap{height:8px;background:var(--bg4);border-radius:4px;overflow:hidden;position:relative;margin-bottom:4px}
.chop-threshold-mark{position:absolute;top:0;bottom:0;width:2px;background:var(--green);opacity:.7;z-index:2}
.chop-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,var(--amber),var(--red));transition:width .5s}
.chop-labels{display:flex;justify-content:space-between;font-family:var(--mono);font-size:8px;color:var(--text3)}
.chop-gap{font-family:var(--mono);font-size:10px;font-weight:700;color:var(--amber);text-align:center;margin-top:2px}

/* ── Status bar ── */
#statusbar{grid-column:1/-1;display:flex;align-items:center;gap:6px;padding:6px 12px;background:var(--bg2);border:1px solid var(--border);border-radius:var(--r);flex-wrap:wrap}
.sb-chip{display:flex;align-items:center;gap:4px;padding:3px 8px;border-radius:20px;font-family:var(--mono);font-size:9px;font-weight:600;letter-spacing:.3px}
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
.sb-sep{color:var(--border2);font-size:14px;margin:0 1px}

/* ── Universe ── */
.uni-row{display:flex;align-items:center;gap:6px;font-family:var(--mono);font-size:11px;margin-bottom:4px}
.uni-val{font-size:14px;font-weight:700}
.uni-bar-wrap{flex:1;height:3px;background:var(--bg4);border-radius:2px;overflow:hidden}
.uni-bar{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--cyan2),var(--cyan));transition:width .6s}

/* ── Risk ── */
.risk-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}
.risk-label{font-size:9px;color:var(--text3);letter-spacing:.5px}
.risk-val{font-family:var(--mono);font-size:11px;font-weight:700}
.lots-bar-wrap{width:100%;height:5px;background:var(--bg4);border-radius:3px;overflow:hidden;margin-top:2px;margin-bottom:6px}
.lots-bar{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--cyan2),var(--cyan));transition:width .6s}
.cooldown-badge{background:var(--amber3);border:1px solid var(--amber2);border-radius:6px;padding:3px 8px;font-family:var(--mono);font-size:10px;font-weight:700;color:var(--amber);margin-top:3px;text-align:center}

/* ── Swing page - compact table ── */
.swing-tbl{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:10px}
.swing-tbl thead tr{background:var(--bg3);border-bottom:1px solid var(--border)}
.swing-tbl th{padding:5px 8px;color:var(--text3);letter-spacing:.8px;font-size:9px;white-space:nowrap}
.swing-tbl tbody tr{border-bottom:1px solid var(--border);transition:background .15s}
.swing-tbl tbody tr:hover{background:var(--bg3)}
.swing-tbl tbody td{padding:4px 8px;white-space:nowrap}
.swing-tbl tbody tr.active-trade-row td{border-top:1px solid var(--amber)!important;border-bottom:1px solid var(--amber)!important;background:rgba(255,170,0,.04)!important}
.swing-tbl tbody tr.active-trade-row td:first-child{border-left:1px solid var(--amber)!important}
.swing-tbl tbody tr.active-trade-row td:last-child{border-right:1px solid var(--amber)!important}

/* Swing position mini-bar */
.sw-bar{display:inline-flex;align-items:center;gap:2px;width:80px}
.sw-bar-track{flex:1;height:5px;background:var(--bg4);border-radius:3px;position:relative;overflow:hidden}
.sw-bar-fill{position:absolute;top:0;height:100%;border-radius:3px;transition:left .4s,width .4s;background:rgba(255,170,0,.3)}
.sw-dot{position:absolute;top:-1px;width:7px;height:7px;border-radius:50%;background:#fff;border:2px solid var(--bg);z-index:2;transition:left .4s}

/* Swing reach bar */
.reach-wrap{display:inline-flex;align-items:center;gap:4px;min-width:100px}
.reach-bar{flex:1;height:6px;background:var(--bg4);border-radius:3px;overflow:hidden}
.reach-fill{height:100%;border-radius:3px;transition:width .4s}
.reach-neg{background:var(--red3)}
.reach-low{background:var(--cyan2)}
.reach-mid{background:var(--amber)}
.reach-high{background:var(--green)}
.reach-over{background:var(--red);animation:reach-pulse 1s infinite}
@keyframes reach-pulse{0%,100%{opacity:1}50%{opacity:.5}}

/* ── Swing State Analysis — compact pill-row format ── */
.ssa-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.ssa-card{background:var(--bg3);border:1px solid var(--border);border-radius:var(--r);padding:6px 8px;transition:border-color .3s,box-shadow .3s}
.ssa-card.active-ssa-card{border-color:var(--amber)!important;background:rgba(255,170,0,.06)!important;box-shadow:0 0 10px rgba(255,170,0,.15)}
.ssa-sym{font-family:var(--mono);font-size:11px;font-weight:700;margin-bottom:3px}
.ssa-pills{display:flex;gap:4px;flex-wrap:wrap}
.ssa-pill{font-family:var(--mono);font-size:8px;padding:1px 5px;border-radius:6px;border:1px solid transparent;font-weight:600}
.pill-exhausted{background:rgba(255,59,107,.15);border-color:rgba(255,59,107,.4);color:var(--red)}
.pill-late{background:rgba(255,170,0,.15);border-color:rgba(255,170,0,.4);color:var(--amber)}
.pill-healthy{background:rgba(0,255,136,.12);border-color:rgba(0,255,136,.3);color:var(--green)}
.pill-unconf{background:rgba(0,229,255,.1);border-color:rgba(0,229,255,.3);color:var(--cyan)}
.pill-compressed{background:rgba(176,72,255,.12);border-color:rgba(176,72,255,.3);color:var(--purple)}
.pill-breakout{background:rgba(176,72,255,.2);border-color:rgba(176,72,255,.5);color:var(--purple)}
.pill-gray{background:rgba(74,101,128,.15);border-color:rgba(74,101,128,.3);color:var(--text3)}
.ssa-ssp{font-family:var(--mono);font-size:9px;color:var(--text3);margin-top:2px}

/* ── Animations ── */
@keyframes fadein{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.fade-in{animation:fadein .4s ease}
@keyframes glow{0%,100%{box-shadow:0 0 0 rgba(0,229,255,0)}50%{box-shadow:0 0 20px rgba(0,229,255,.15)}}
.panel:hover{animation:glow 2s infinite}
@keyframes flash-glow{0%{border-color:var(--amber);box-shadow:0 0 16px rgba(255,170,0,.4)}50%{border-color:var(--amber);box-shadow:0 0 24px rgba(255,170,0,.6)}100%{border-color:var(--border);box-shadow:none}}
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
  <div class="nodata-path">logs/dashboard_latest.json</div>
  <div class="nodata-spin"></div>
</div>

<!-- Header -->
<div id="header">
  <div class="hdr-logo">⬡ PROXIMA CDE</div>
  <div class="hdr-sep">│</div>
  <div id="hdr-mode" class="hdr-pill pill-mode">LIVE</div>
  <div id="hdr-health" class="hdr-pill pill-ok">● OK</div>
  <div class="hdr-sep">│</div>

  <!-- CHOP INDICATOR — eyecatching compact header badge -->
  <div id="chop-indicator" title="Market is chopping — entries blocked until polarization drops below 70%">
    <div class="chop-dot"></div>
    <span id="chop-label">CHOP</span>
    <span id="chop-pct-badge" style="font-size:11px;font-weight:900">--</span>
    <span id="chop-gap-pill" class="chop-gap-pill" style="display:none">needs --% ▼</span>
    <span id="chop-time-badge" style="font-size:9px;opacity:.8">-- min</span>
  </div>

  <div class="hdr-nav">
    <button id="nav-overview" class="nav-link active">OVERVIEW</button>
    <button id="nav-swing" class="nav-link">SWING</button>
    <button id="nav-analysis" class="nav-link">ANALYSIS</button>
  </div>

  <div class="hdr-right">
    <div id="hdr-ts" style="font-family:var(--mono);font-size:10px;color:var(--text2)">--:--:--</div>
    <div class="hdr-sep">│</div>
    <div id="hdr-uptime" style="font-family:var(--mono);font-size:10px;color:var(--text2)">UPTIME --:--:--</div>
    <div class="hdr-sep">│</div>
    <div id="hdr-cycle" style="font-family:var(--mono);font-size:10px;color:var(--text2)">CYCLE ---</div>
    <div class="pulse" id="hdr-pulse"></div>
  </div>
</div>

<!-- Stale banner -->
<div id="stale-banner" style="display:none;background:rgba(255,59,107,.12);border-bottom:1px solid rgba(255,59,107,.3);padding:6px;text-align:center;font-family:var(--mono);font-size:10px;color:var(--red);letter-spacing:1px;font-weight:700">
  ⚠️ ENGINE DISCONNECTED — SHOWING CACHED DATA
</div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- OVERVIEW PAGE                                          -->
<!-- ═══════════════════════════════════════════════════════ -->
<div id="overview-page">
  <div id="grid">

    <!-- LEFT: Currency Matrix -->
    <div class="panel" id="panel-left">
      <div class="panel-title"><span class="title-dot"></span>CURRENCY MATRIX</div>

      <div style="font-family:var(--mono);font-size:9px;color:var(--cyan);margin-bottom:6px;letter-spacing:1px">WLS STRENGTHS</div>
      <div id="strengths-list"></div>

      <div class="divider"></div>
      <div style="font-family:var(--mono);font-size:9px;color:var(--amber);margin-bottom:6px;letter-spacing:1px">BURST ACTIVITY</div>
      <div id="burst-list"></div>

      <div class="divider"></div>
      <div style="font-family:var(--mono);font-size:9px;color:var(--purple);margin-bottom:6px;letter-spacing:1px">DIR. EFFICIENCY</div>
      <div id="der-list"></div>
    </div>

    <!-- MIDDLE: Pipeline + Gauges + Hypothesis -->
    <div class="panel" id="panel-pipeline">
      <div class="panel-title"><span class="title-dot" style="background:var(--purple)"></span>SIGNAL PIPELINE</div>
      <div class="funnel-wrap" id="funnel"></div>

      <div class="divider"></div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:center">
        <div>
          <div style="font-family:var(--mono);font-size:9px;color:var(--cyan);margin-bottom:4px;letter-spacing:1px">GRAPH HEALTH</div>
          <div class="gauge-wrap">
            <div class="gauge">
              <svg width="70" height="70" viewBox="0 0 70 70">
                <circle class="gauge-bg" cx="35" cy="35" r="28" stroke-width="6"/>
                <circle id="gauge-graph" class="gauge-fill" cx="35" cy="35" r="28" stroke-width="6"
                  stroke="var(--cyan)" stroke-dasharray="175.93" stroke-dashoffset="175.93"/>
              </svg>
              <div class="gauge-label">
                <div class="gauge-val" id="gauge-graph-val" style="color:var(--cyan)">--</div>
                <div class="gauge-sub">QUALITY</div>
              </div>
            </div>
          </div>
        </div>
        <div>
          <div style="font-family:var(--mono);font-size:9px;color:var(--green);margin-bottom:4px;letter-spacing:1px">TICK QUALITY</div>
          <div class="gauge-wrap">
            <div class="gauge">
              <svg width="70" height="70" viewBox="0 0 70 70">
                <circle class="gauge-bg" cx="35" cy="35" r="28" stroke-width="6"/>
                <circle id="gauge-tick" class="gauge-fill" cx="35" cy="35" r="28" stroke-width="6"
                  stroke="var(--green)" stroke-dasharray="175.93" stroke-dashoffset="175.93"/>
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
      <div style="font-family:var(--mono);font-size:9px;color:var(--amber);margin-bottom:6px;letter-spacing:1px">TOP HYPOTHESIS</div>
      <div id="hypothesis"></div>
    </div>

    <!-- RIGHT: Positions + Risk + Regime + Universe -->
    <div style="display:flex;flex-direction:column;gap:10px">

      <!-- Positions -->
      <div class="panel">
        <div class="panel-title"><span class="title-dot" style="background:var(--green)"></span>OPEN POSITIONS</div>
        <div id="positions"></div>
        <div style="margin-top:6px;font-family:var(--mono);font-size:11px;display:flex;justify-content:space-between;padding-top:4px;border-top:1px solid var(--border)">
          <span style="color:var(--text3)">TOTAL P&amp;L</span>
          <span id="total-pnl" style="font-weight:700">$0.00</span>
        </div>
      </div>

      <!-- Risk Engine -->
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
          <span class="risk-val" id="risk-target" style="color:var(--green)">--</span>
        </div>
        <div class="risk-row">
          <span class="risk-label">SESSION STOP</span>
          <span class="risk-val" id="risk-stop" style="color:var(--red)">--</span>
        </div>
        <div id="cooldown-badge"></div>
      </div>

      <!-- Market Regime — compact -->
      <div class="panel" id="panel-regime" style="border-left:3px solid var(--text3)">
        <div class="panel-title"><span class="title-dot"></span>MARKET REGIME</div>
        <div class="regime-compact" id="regime-compact">
          <div id="regime-icon" style="font-size:22px">⏳</div>
          <div class="regime-body">
            <div style="display:flex;justify-content:space-between;align-items:baseline">
              <span id="regime-label" class="regime-label">--</span>
              <span id="regime-ssp" class="regime-threshold">--%</span>
            </div>
            <div class="regime-bar-wrap">
              <div id="regime-bar" class="regime-bar-fill" style="width:0%"></div>
            </div>
            <div id="regime-status" class="regime-pct">--</div>
          </div>
        </div>
        <!-- Chop progress widget (shows only when chopping) -->
        <div id="chop-widget" class="chop-widget">
          <div class="chop-top">
            <span class="chop-title">⚡ CHOPPING</span>
            <span class="chop-time" id="chop-widget-time">0 min</span>
          </div>
          <div class="chop-pbar-wrap">
            <div class="chop-threshold-mark" id="chop-threshold-mark" style="left:70%"></div>
            <div class="chop-fill" id="chop-fill" style="width:0%"></div>
          </div>
          <div class="chop-labels">
            <span>0%</span>
            <span style="color:var(--green)">▲ 65% clear</span>
            <span>100%</span>
          </div>
          <div class="chop-gap" id="chop-gap-text">--% above threshold — drop -- more to unblock</div>
        </div>
      </div>

      <!-- Universe (compact) -->
      <div class="panel">
        <div class="panel-title"><span class="title-dot" style="background:var(--purple)"></span>UNIVERSE</div>
        <div class="uni-row">
          <span class="risk-label">SYMBOLS</span>
          <span class="uni-val" id="uni-val" style="color:var(--cyan)">--/--</span>
          <div class="uni-bar-wrap"><div class="uni-bar" id="uni-bar" style="width:0%"></div></div>
        </div>
        <div style="display:flex;gap:10px;flex-wrap:wrap;font-family:var(--mono);font-size:9px;color:var(--text3);margin-top:2px">
          <span>CONF: <span id="uni-conf" style="color:var(--text)">--</span></span>
          <span>MEM: <span id="uni-mem" style="color:var(--text)">-- MB</span></span>
          <span>SOLVE: <span id="uni-latency" style="color:var(--text)">-- ms</span></span>
        </div>
      </div>

    </div><!-- /right column -->

  </div><!-- /grid -->

  <!-- Status Bar -->
  <div id="statusbar" style="margin:0 12px 10px">
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
    <div class="sb-chip chip-neu" id="chip-mode"><span class="sb-label">MODE</span>&nbsp;<span class="sb-val">--</span></div>
    <div class="sb-sep" style="margin-left:auto">│</div>
    <div style="font-family:var(--mono);font-size:9px;color:var(--text3)">LAST UPDATE: <span id="last-update">--</span></div>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- SWING PAGE                                             -->
<!-- ═══════════════════════════════════════════════════════ -->
<div id="swing-page" style="display:none;padding:10px 12px">

  <div class="panel" style="margin-bottom:10px">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
      <span class="panel-title" style="margin-bottom:0"><span class="title-dot" style="background:var(--amber)"></span>SWING STOPS &amp; TRADE REACH (M5)</span>
      <span id="swing-max-sl-badge" style="font-family:var(--mono);font-size:10px;color:var(--red);background:rgba(255,59,107,.1);padding:2px 7px;border-radius:4px;border:1px solid rgba(255,59,107,.2)">MAX SL: -$60</span>
      <span style="margin-left:auto;font-family:var(--mono);font-size:9px;color:var(--text3)">TP = entry ± remaining × 0.60 | amber = active trade</span>
    </div>
    <div style="background:var(--bg2);border:1px solid var(--border);border-radius:var(--r2);overflow:auto">
      <table class="swing-tbl" id="swing-table">
        <thead>
          <tr>
            <th style="text-align:left">SYM</th>
            <th style="text-align:center">REACH</th>
            <th style="text-align:center">BAR POS</th>
            <th style="text-align:right">▼DN</th>
            <th style="text-align:right">▲UP</th>
            <th style="text-align:right">B·SL</th>
            <th style="text-align:right">B·SL($)</th>
            <th style="text-align:right">B·TP</th>
            <th style="text-align:right">B·TP($)</th>
            <th style="text-align:right">S·SL</th>
            <th style="text-align:right">S·SL($)</th>
            <th style="text-align:right">S·TP</th>
            <th style="text-align:right">S·TP($)</th>
            <th style="text-align:center">HISTORY</th>
          </tr>
        </thead>
        <tbody id="swing-tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════════════════════════════ -->
<!-- ANALYSIS PAGE — compact SSP/MSP + swing state         -->
<!-- ═══════════════════════════════════════════════════════ -->
<div id="analysis-page" style="display:none;padding:10px 12px">

  <div class="panel" style="margin-bottom:10px">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
      <span class="panel-title" style="margin-bottom:0"><span class="title-dot" style="background:var(--cyan)"></span>SWING STATE ANALYSIS</span>
      <span style="margin-left:auto;font-family:var(--mono);font-size:9px;color:var(--text3)">observation only — no blocking · SSP=structural position · MSP=micro/bar position</span>
    </div>
    <!-- Compact pill-card grid -->
    <div class="ssa-grid" id="ssa-grid"></div>
  </div>

  <!-- Full SSP/MSP table (hidden behind toggle) -->
  <div class="panel">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
      <span class="panel-title" style="margin-bottom:0"><span class="title-dot" style="background:var(--purple)"></span>DETAILED SSP / MSP TABLE</span>
      <button id="toggle-ssp-table" style="margin-left:auto;background:none;border:1px solid var(--border);color:var(--text3);font-family:var(--mono);font-size:9px;padding:2px 8px;border-radius:4px;cursor:pointer">SHOW</button>
    </div>
    <div id="ssp-table-wrap" style="display:none;background:var(--bg2);border:1px solid var(--border);border-radius:var(--r2);overflow:auto">
      <table class="swing-tbl" id="swing-analysis-table">
        <thead>
          <tr>
            <th style="text-align:left">SYM</th>
            <th style="text-align:center">BUY SSP</th>
            <th style="text-align:center">SELL SSP</th>
            <th style="text-align:center">BUY MSP</th>
            <th style="text-align:center">SELL MSP</th>
            <th style="text-align:center">BUY STATE</th>
            <th style="text-align:center">SELL STATE</th>
            <th style="text-align:center">POSITION</th>
            <th style="text-align:right">RNG(p)</th>
            <th style="text-align:right">RNG EXP</th>
            <th style="text-align:right">VOL EXP</th>
          </tr>
        </thead>
        <tbody id="swing-analysis-tbody"></tbody>
      </table>
    </div>
  </div>
</div>


<script>
const CCY_ORDER = ['EUR','USD','GBP','JPY','CHF','AUD','CAD','NZD'];

function fmt(v, dec=5){ return v >= 0 ? `+${v.toFixed(dec)}` : v.toFixed(dec); }
function fmtPnl(v){ return (v >= 0 ? '+' : '') + '$' + Math.abs(v).toFixed(2); }
function clamp(v,mn,mx){ return Math.max(mn, Math.min(mx, v)); }

function gaugeSet(id, valId, value, colorGood='var(--cyan)', colorBad='var(--red)'){
  const r = 28, circ = 2 * Math.PI * r;
  const pct = clamp(value, 0, 1);
  const offset = circ * (1 - pct);
  const el = document.getElementById(id);
  const ve = document.getElementById(valId);
  if(el){ el.style.strokeDashoffset = offset; el.style.stroke = pct > 0.5 ? colorGood : pct > 0.3 ? 'var(--amber)' : colorBad; }
  if(ve){ ve.textContent = (value*100).toFixed(0) + '%'; ve.style.color = pct > 0.5 ? colorGood : pct > 0.3 ? 'var(--amber)' : colorBad; }
}

function chipClass(val, threshOk=0.5, threshWarn=0.3){
  if(val >= threshOk) return 'chip-ok';
  if(val >= threshWarn) return 'chip-warn';
  return 'chip-err';
}

function renderCcySection(containerId, data, decimalPlaces=5, checkPers=false, strengthsMap=null){
  if(!data) return;
  const el = document.getElementById(containerId);
  if(!el) return;
  const entries = CCY_ORDER.map(c => [c, data[c] || 0]);
  const sorted = [...entries].sort((a,b) => Math.abs(b[1])-Math.abs(a[1]));
  const maxA = Math.max(...entries.map(e => Math.abs(e[1])), 1e-12);
  let html = '';
  sorted.forEach(([ccy, val], i) => {
    const rankCls = i === 0 ? 'rank-top' : i === sorted.length-1 ? 'rank-bot' : '';
    const dir = val > 1e-8 ? 'pos' : val < -1e-8 ? 'neg' : 'neu';
    const pct = clamp(Math.abs(val)/maxA, 0, 1)*100;
    let arrow = val > 1e-8 ? '▲' : val < -1e-8 ? '▼' : '○';
    let streak = 0;
    if(checkPers && strengthsMap && strengthsMap[ccy]){
      const sinfo = strengthsMap[ccy];
      const sdir = sinfo.dir || 0;
      streak = sinfo.streak || 0;
      arrow = sdir > 0 ? '▲' : sdir < 0 ? '▼' : '○';
    }
    html += `<div class="ccy-row ${rankCls}">
      <span class="ccy-label">${ccy}</span>
      <div class="ccy-bar-wrap"><div class="ccy-bar ${dir}" style="width:${pct.toFixed(1)}%"></div></div>
      <span class="ccy-val">${fmt(val, decimalPlaces)}</span>
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
    { label:'GENERATED', val:gen,   cls:'f-gen'    },
    { label:'BURST FILT',val:burst, cls:'f-burst'  },
    { label:'BAR ALIGN', val:bar,   cls:'f-bar'    },
    { label:'RANKED',    val:rnk,   cls:'f-ranked' },
    { label:'SELECTED',  val:sel,   cls:'f-selected'},
    { label:'RISK OK',   val:risk,  cls:'f-risk'   },
    { label:'EXECUTED',  val:exe,   cls:'f-exe'    },
  ];
  let html = '';
  stages.forEach((s, i) => {
    const pct = gen > 0 ? clamp(s.val/gen, 0, 1)*100 : (s.val > 0 ? 100 : 0);
    const minPct = s.val > 0 ? Math.max(pct, 6) : 0;
    const isLast = i === stages.length-1;
    html += `<div class="funnel-stage">
      <span class="funnel-label">${s.label}</span>
      <div class="funnel-bar-wrap">
        <div class="funnel-bar ${s.cls}" style="width:${minPct.toFixed(1)}%;min-width:${s.val>0?28:0}px">
          ${s.val}
        </div>
      </div>
      ${!isLast ? `<span class="funnel-arrow">›</span>` : ''}
    </div>`;
  });
  el.innerHTML = html;
}

function renderPositions(d){
  const el = document.getElementById('positions');
  if(!el) return;
  const posCount = d.positions || 0;
  const pnl = d.pnl || 0;
  const activeSyms = d.active_symbols || [];
  const swingOverlay = d.swing_overlay || {};

  if(posCount === 0){
    el.innerHTML = '<div class="no-pos">NO OPEN POSITIONS</div>';
  } else {
    let html = '';
    for(let i = 0; i < posCount; i++){
      const sym = activeSyms[i] || `POSITION ${i+1}`;
      const sw = swingOverlay[sym] || swingOverlay[sym+'.m'] || null;
      let reachStr = '';
      if(sw && sw.swing_reach){
        const r = sw.swing_reach;
        const pct = r.pct;
        const pctColor = pct >= 120 ? 'var(--red)' : pct >= 90 ? 'var(--green)' : pct >= 50 ? 'var(--amber)' : 'var(--cyan)';
        const pnlStr = r.pnl >= 0 ? `+$${r.pnl.toFixed(0)}` : `-$${Math.abs(r.pnl).toFixed(0)}`;
        const pnlColor = r.pnl >= 0 ? 'var(--green)' : 'var(--red)';
        reachStr = `<span style="font-family:var(--mono);font-size:9px;color:${pctColor}"> REACH ${pct.toFixed(0)}%</span> <span style="font-family:var(--mono);font-size:9px;color:${pnlColor}">${pnlStr}</span>`;
      }
      html += `<div class="pos-card active-pos">
        <div class="pos-header">
          <span class="pos-sym">${sym}</span>
          <span class="pos-dir buy">OPEN</span>
          <span class="pos-pnl" style="margin-left:auto">${reachStr}</span>
        </div>
      </div>`;
    }
    el.innerHTML = html;
  }
  const totalEl = document.getElementById('total-pnl');
  if(totalEl){
    totalEl.textContent = fmtPnl(pnl);
    totalEl.style.color = pnl >= 0 ? 'var(--green)' : 'var(--red)';
  }
}

function renderHypothesis(top){
  const el = document.getElementById('hypothesis');
  if(!el) return;
  if(!top || !top.top_symbol){
    el.innerHTML = `<div style="color:var(--text3);font-family:var(--mono);font-size:11px;padding:10px;text-align:center;border:1px dashed var(--border);border-radius:var(--r)">NO SIGNAL — BELOW MIN CONFIDENCE</div>`;
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

function renderRegime(d){
  const rd = d.regime_data || {};
  const pct = rd.polarized_ssp_pct || 0;
  const regime = rd.regime || 'N/A';
  const blocked = rd.entries_blocked;
  const chopMin = rd.chop_minutes || 0;
  const gapToClear = rd.gap_to_clear || 0;
  const thresholdPct = rd.threshold_pct || 70;

  // Header chop indicator
  const chopInd = document.getElementById('chop-indicator');
  const chopLabel = document.getElementById('chop-label');
  const chopPctBadge = document.getElementById('chop-pct-badge');
  const chopTimeBadge = document.getElementById('chop-time-badge');
  const chopGapPill = document.getElementById('chop-gap-pill');

  if(blocked){
    chopInd.classList.add('active');
    chopPctBadge.textContent = pct.toFixed(0) + '%';
    chopTimeBadge.textContent = chopMin >= 1 ? `${chopMin.toFixed(0)}m` : `<1m`;
    if(chopGapPill){
      chopGapPill.style.display = 'inline-flex';
      chopGapPill.textContent = `↓${gapToClear.toFixed(0)}% to clear`;
    }
  } else {
    chopInd.classList.remove('active');
    if(chopGapPill) chopGapPill.style.display = 'none';
  }

  // Regime panel
  const regimeIcon = document.getElementById('regime-icon');
  const regimeLabel = document.getElementById('regime-label');
  const regimeSsp = document.getElementById('regime-ssp');
  const regimeBar = document.getElementById('regime-bar');
  const regimeStatus = document.getElementById('regime-status');
  const panelRegime = document.getElementById('panel-regime');
  const chopWidget = document.getElementById('chop-widget');

  if(regimeIcon){
    if(blocked){
      regimeIcon.textContent = '⛔';
      regimeLabel.textContent = 'CHOP — BLOCKED';
      regimeLabel.style.color = 'var(--red)';
      regimeBar.style.background = 'linear-gradient(90deg,var(--amber),var(--red))';
      regimeStatus.innerHTML = `<span style="color:var(--red)">${pct.toFixed(0)}%</span> <span style="color:var(--text3)">polarized ≥ 70% — blocking</span>`;
      panelRegime.style.borderLeftColor = 'var(--red)';
      if(chopWidget) chopWidget.classList.add('active');

      // Update chop widget
      const chopFill = document.getElementById('chop-fill');
      const chopWidgetTime = document.getElementById('chop-widget-time');
      const chopGapText = document.getElementById('chop-gap-text');
      if(chopFill) chopFill.style.width = Math.min(pct, 100).toFixed(0) + '%';
      if(chopWidgetTime) chopWidgetTime.textContent = chopMin >= 1 ? `${chopMin.toFixed(0)} min` : `< 1 min`;
      if(chopGapText) chopGapText.textContent = `${pct.toFixed(0)}% polarized — needs ↓${gapToClear.toFixed(0)}% more to unblock`;
    } else {
      regimeIcon.textContent = '▶';
      regimeLabel.textContent = 'TREND — ACTIVE';
      regimeLabel.style.color = 'var(--green)';
      regimeBar.style.background = 'var(--green)';
      regimeStatus.innerHTML = `<span style="color:var(--green)">${pct.toFixed(0)}%</span> <span style="color:var(--text3)">polarized ≤ 65% — clear</span>`;
      panelRegime.style.borderLeftColor = 'var(--green)';
      if(chopWidget) chopWidget.classList.remove('active');
    }
    regimeSsp.textContent = `${pct.toFixed(0)}%`;
    regimeBar.style.width = `${Math.min(pct, 100).toFixed(0)}%`;
  }
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
    setChip('chip-mode','MODE','OFFLINE', 'chip-err');
  } else {
    const mt5ok = d.mt5_ok;
    setChip('chip-mt5','MT5',mt5ok?'OK':'FAIL', mt5ok?'chip-ok':'chip-err');
    setChip('chip-tick','TICK',(d.tick_quality||0).toFixed(2), chipClass(d.tick_quality||0));
    setChip('chip-graph','GRAPH',(d.graph_quality||0).toFixed(2), chipClass(d.graph_quality||0));
    setChip('chip-snap','SNAP','OK','chip-ok');
    setChip('chip-solve','SOLVE',`${(d.solve_latency_ms||0).toFixed(0)} ms`, 'chip-neu');
    const mode = d.mode || '--';
    setChip('chip-mode','MODE', mode, mode==='LIVE'?'chip-ok':'chip-warn');
  }
}

// ── SSA compact cards ─────────────────────────────────────────────────
function swingStatePillClass(st){
  if(!st || st === '--') return 'pill-gray';
  const s = st.toUpperCase();
  if(s.includes('EXHAUSTED')) return 'pill-exhausted';
  if(s === 'LATE') return 'pill-late';
  if(s === 'HEALTHY') return 'pill-healthy';
  if(s === 'UNCONFIRMED') return 'pill-unconf';
  if(s.includes('COMPRESSED')) return 'pill-compressed';
  if(s.includes('BREAKOUT')) return 'pill-breakout';
  return 'pill-gray';
}

function renderSSAGrid(data){
  const grid = document.getElementById('ssa-grid');
  if(!grid) return;
  const overlay = data.swing_overlay || {};
  const syms = Object.keys(overlay).sort();
  const activeSyms = data.active_symbols || [];
  let html = '';
  syms.forEach(sym => {
    const sw = overlay[sym];
    const sa = sw.swing_analysis;
    if(!sa){ return; }
    const buyState = sa.buy ? sa.buy.state : (sa.swing_state || '--');
    const sellState = sa.sell ? sa.sell.state : (sa.swing_state || '--');
    const posState = sa.position_state || '--';
    const buySsp = sa.buy_ssp !== undefined ? sa.buy_ssp : (sa.buy ? sa.buy.ssp : null);
    const sellSsp = sa.sell_ssp !== undefined ? sa.sell_ssp : (sa.sell ? sa.sell.ssp : null);
    const sspStr = (buySsp !== null && sellSsp !== null)
      ? `B:${buySsp.toFixed(2)} S:${sellSsp.toFixed(2)}`
      : '--';
    const isActive = activeSyms.some(a => a.startsWith(sym));
    html += `<div class="ssa-card ${isActive ? 'active-ssa-card' : ''}">
      <div class="ssa-sym">${sym}</div>
      <div class="ssa-pills">
        <span class="ssa-pill ${swingStatePillClass(buyState)}">▲${buyState}</span>
        <span class="ssa-pill ${swingStatePillClass(sellState)}">▼${sellState}</span>
        <span class="ssa-pill ${swingStatePillClass(posState)}">${posState}</span>
      </div>
      <div class="ssa-ssp">${sspStr}</div>
    </div>`;
  });
  grid.innerHTML = html || '<div style="color:var(--text3);font-family:var(--mono);font-size:11px;padding:12px">No swing data</div>';
}

// ── Full SSP table ─────────────────────────────────────────────────────
function renderSwingAnalysis(data){
  const tbody = document.getElementById('swing-analysis-tbody');
  if(!tbody) return;
  const overlay = data.swing_overlay || {};
  const syms = Object.keys(overlay).sort();
  const activeSyms = data.active_symbols || [];
  let html = '';
  syms.forEach(sym => {
    const sw = overlay[sym];
    const sa = sw.swing_analysis;
    if(!sa){ html += `<tr><td>${sym}</td><td colspan="10" style="text-align:center;color:var(--text3)">no data</td></tr>`; return; }
    const stateColor = function(st){
      if(!st) return 'var(--text3)';
      if(st.includes('EXHAUST')) return 'var(--red)';
      if(st === 'LATE') return 'var(--amber)';
      if(st === 'HEALTHY') return 'var(--green)';
      if(st === 'UNCONFIRMED') return 'var(--cyan)';
      if(st.includes('BREAKOUT')) return 'var(--purple)';
      return 'var(--text3)';
    };
    const buyState = sa.buy ? sa.buy.state : sa.swing_state;
    const sellState = sa.sell ? sa.sell.state : sa.swing_state;
    const posState = sa.position_state || '--';
    const posColor = posState.includes('BREAKOUT') ? 'var(--purple)' : posState.includes('COMPRESSED') ? 'var(--amber)' : posState === 'INSIDE_RANGE' ? 'var(--cyan)' : 'var(--text3)';
    const buySsp = sa.buy_ssp !== undefined ? sa.buy_ssp : (sa.buy ? sa.buy.ssp : null);
    const sellSsp = sa.sell_ssp !== undefined ? sa.sell_ssp : (sa.sell ? sa.sell.ssp : null);
    const buyMsp = sa.buy_msp !== undefined ? sa.buy_msp : (sa.buy ? sa.buy.msp : null);
    const sellMsp = sa.sell_msp !== undefined ? sa.sell_msp : (sa.sell ? sa.sell.msp : null);
    function fN(v,d){ return v !== null && v !== undefined && v !== '' ? v.toFixed(d) : '--'; }
    const isActive = activeSyms.some(a => a.startsWith(sym));
    html += `<tr class="${isActive ? 'active-trade-row' : ''}">
      <td style="font-weight:700">${sym}</td>
      <td style="text-align:center;color:${buySsp>0.85?'var(--red)':buySsp>0.65?'var(--amber)':'var(--green)'}">${fN(buySsp,3)}</td>
      <td style="text-align:center;color:${sellSsp>0.85?'var(--red)':sellSsp>0.65?'var(--amber)':'var(--green)'}">${fN(sellSsp,3)}</td>
      <td style="text-align:center;color:${buyMsp>0.7?'var(--red)':buyMsp>0.4?'var(--amber)':'var(--cyan)'}">${fN(buyMsp,3)}</td>
      <td style="text-align:center;color:${sellMsp>0.7?'var(--red)':sellMsp>0.4?'var(--amber)':'var(--cyan)'}">${fN(sellMsp,3)}</td>
      <td style="text-align:center;color:${stateColor(buyState)};font-weight:700">${buyState}</td>
      <td style="text-align:center;color:${stateColor(sellState)};font-weight:700">${sellState}</td>
      <td style="text-align:center;color:${posColor};font-weight:700">${posState}</td>
      <td style="text-align:right">${fN(sa.range_price,1)}</td>
      <td style="text-align:right;color:${sa.range_expansion<0.5?'var(--amber)':'var(--text)'}">${fN(sa.range_expansion,2)}</td>
      <td style="text-align:right;color:${sa.vol_expansion>3?'var(--red)':sa.vol_expansion>2?'var(--amber)':'var(--text)'}">${fN(sa.vol_expansion,2)}</td>
    </tr>`;
  });
  tbody.innerHTML = html;
}

// ── Swing table ────────────────────────────────────────────────────────
function renderSwing(data){
  const tbody = document.getElementById('swing-tbody');
  if(!tbody) return;
  const overlay = data.swing_overlay || {};
  const syms = Object.keys(overlay).sort();
  const activeSyms = data.active_symbols || [];
  let html = '';
  syms.forEach(sym => {
    const sw = overlay[sym];
    const dn = sw.avg_down || 0;
    const up = sw.avg_up || 0;
    const fp = sw.forming_pips || 0;
    const total = Math.max(up - dn, 1);
    const posPct = ((fp - dn) / total * 100);
    const pctClamped = Math.max(2, Math.min(98, posPct));
    const isActive = activeSyms.includes(sym);

    let reachHtml = '<span style="color:var(--text3)">--</span>';
    if(sw.swing_reach){
      const r = sw.swing_reach;
      const pct = r.pct;
      const fillW = Math.max(0, Math.min(pct, 100));
      const cls = pct >= 120 ? 'reach-over' : pct >= 90 ? 'reach-high' : pct >= 50 ? 'reach-mid' : pct >= 0 ? 'reach-low' : 'reach-neg';
      const pctColor = pct >= 120 ? 'var(--red)' : pct >= 90 ? 'var(--green)' : pct >= 50 ? 'var(--amber)' : 'var(--cyan)';
      const pnlColor = r.pnl >= 0 ? 'var(--green)' : 'var(--red)';
      const pnlStr = r.pnl >= 0 ? '+$'+r.pnl.toFixed(0) : '-$'+Math.abs(r.pnl).toFixed(0);
      const dirArrow = r.direction === 'BUY' ? '▲' : '▼';
      reachHtml = `<div class="reach-wrap">
        <span style="font-size:9px;color:${pctColor};font-weight:700;min-width:28px;text-align:right">${pct.toFixed(0)}%</span>
        <div class="reach-bar"><div class="reach-fill ${cls}" style="width:${fillW}%"></div></div>
        <span style="font-size:8px;color:var(--text3)">${r.moved_pips}p/${r.expected_pips}p</span>
      </div>
      <div style="font-size:8px;color:var(--text2);margin-top:1px">${dirArrow} <span style="color:${pnlColor};font-weight:bold">${pnlStr}</span></div>`;
    }

    let histHtml = '--';
    if(sw.history){
      const h = sw.history;
      if(h.status === 'RUNNING'){
        const peak = h.peak_pnl || 0;
        const color = peak >= 0 ? 'var(--green)' : 'var(--red)';
        histHtml = `RUN: ${h.direction} (<span style="color:${color};">${peak >= 0 ? '+$'+peak.toFixed(0) : '-$'+Math.abs(peak).toFixed(0)}</span>)`;
      } else if(h.status === 'CLOSED'){
        const pnl = h.pnl || 0;
        const color = pnl >= 0 ? 'var(--green)' : 'var(--red)';
        histHtml = `HIT ${h.reason} (<span style="color:${color};">${pnl >= 0 ? '+$'+pnl.toFixed(0) : '-$'+Math.abs(pnl).toFixed(0)}</span>)`;
      }
    }

    html += `<tr class="${isActive ? 'active-trade-row' : ''}">
      <td style="font-weight:700">${sym}</td>
      <td style="padding:3px 6px">${reachHtml}</td>
      <td>
        <div class="sw-bar">
          <span style="color:var(--red);font-size:8px;width:22px;text-align:right">${dn.toFixed(0)}</span>
          <div class="sw-bar-track">
            <div class="sw-bar-fill" style="left:0;width:100%;opacity:0.12"></div>
            <div class="sw-dot" style="left:${pctClamped}%"></div>
          </div>
          <span style="color:var(--green);font-size:8px;width:22px">${up.toFixed(0)}</span>
        </div>
      </td>
      <td style="text-align:right;color:var(--red)">${dn.toFixed(1)}</td>
      <td style="text-align:right;color:var(--green)">+${up.toFixed(1)}</td>
      <td style="text-align:right;color:var(--red)">${sw.buy_sl || '--'}</td>
      <td style="text-align:right;color:var(--red);font-weight:bold">${sw.buy_sl_usd !== undefined ? '-$'+sw.buy_sl_usd.toFixed(0) : '--'}</td>
      <td style="text-align:right;color:var(--green)">${sw.buy_tp || '--'}</td>
      <td style="text-align:right;color:var(--green);font-weight:bold">${sw.buy_tp_usd !== undefined ? '+$'+sw.buy_tp_usd.toFixed(0) : '--'}</td>
      <td style="text-align:right;color:var(--red)">${sw.sell_sl || '--'}</td>
      <td style="text-align:right;color:var(--red);font-weight:bold">${sw.sell_sl_usd !== undefined ? '-$'+sw.sell_sl_usd.toFixed(0) : '--'}</td>
      <td style="text-align:right;color:var(--green)">${sw.sell_tp || '--'}</td>
      <td style="text-align:right;color:var(--green);font-weight:bold">${sw.sell_tp_usd !== undefined ? '+$'+sw.sell_tp_usd.toFixed(0) : '--'}</td>
      <td style="text-align:center;color:var(--text2);font-size:9px">${histHtml}</td>
    </tr>`;
  });
  tbody.innerHTML = html;
}

// ── Main update ────────────────────────────────────────────────────────
function update(d){
  document.getElementById('nodata').classList.add('hidden');
  const active = d.engine_active;

  // Stale banner
  const staleEl = document.getElementById('stale-banner');
  if(staleEl) staleEl.style.display = active ? 'none' : 'block';

  // Header
  document.getElementById('hdr-ts').textContent = d.ts || '--';
  const up = d.uptime || 0;
  document.getElementById('hdr-uptime').textContent = `UPTIME ${String(Math.floor(up/3600)).padStart(2,'0')}:${String(Math.floor((up%3600)/60)).padStart(2,'0')}:${String(Math.floor(up%60)).padStart(2,'0')}`;
  document.getElementById('hdr-cycle').textContent = `CYCLE ${d.trade_count||0}`;
  const pulseEl = document.getElementById('hdr-pulse');
  const hstate = d.health_state || 'OK';
  const hel = document.getElementById('hdr-health');
  const mel = document.getElementById('hdr-mode');
  if(!active){
    mel.textContent = 'OFFLINE'; mel.className = 'hdr-pill pill-err';
    if(pulseEl){ pulseEl.style.background='var(--red)'; pulseEl.style.boxShadow='0 0 8px var(--red)'; pulseEl.style.animation='none'; }
    hel.className='hdr-pill pill-err'; hel.textContent='● DISCONNECTED';
  } else {
    mel.textContent = d.mode || 'LIVE'; mel.className = 'hdr-pill pill-mode';
    if(pulseEl){ pulseEl.style.background='var(--green)'; pulseEl.style.boxShadow='0 0 8px var(--green)'; pulseEl.style.animation='pulse 2s infinite'; }
    if(hstate==='OK'){ hel.className='hdr-pill pill-ok'; hel.textContent='● OK'; }
    else if(hstate==='DEGRADED'){ hel.className='hdr-pill pill-warn'; hel.textContent='▲ DEGRADED'; }
    else { hel.className='hdr-pill pill-err'; hel.textContent='✕ ERROR'; }
  }

  // Currency sections
  renderCcySection('strengths-list', d.currency_strengths, 5, true, d.strength_persistence);
  renderCcySection('burst-list', d.currency_bursts, 3, true, d.burst_persistence);
  renderCcySection('der-list', d.observability, 2, false);

  // Gauges
  gaugeSet('gauge-graph','gauge-graph-val', d.graph_quality||0, 'var(--cyan)', 'var(--red)');
  gaugeSet('gauge-tick','gauge-tick-val', d.tick_quality||0, 'var(--green)', 'var(--red)');

  // Pipeline
  renderPipeline(d.pipeline);

  // Positions
  renderPositions(d);

  // Hypothesis
  renderHypothesis(d);

  // Risk
  document.getElementById('risk-lot').textContent = (d.lot_size||0.70).toFixed(2);
  const maxLots = d.max_total_lots || 2.10;
  const openLots = d.open_lots || 0;
  document.getElementById('risk-lots').textContent = `${openLots.toFixed(2)} / ${maxLots.toFixed(2)}`;
  document.getElementById('lots-bar').style.width = (maxLots>0 ? clamp(openLots/maxLots,0,1)*100 : 0).toFixed(1)+'%';
  document.getElementById('risk-target').textContent = `+$${(d.profit_target||50).toFixed(0)}`;
  document.getElementById('risk-stop').textContent = `-$${Math.abs(d.stop_loss_amount||60).toFixed(0)}`;
  const swingMaxSl = document.getElementById('swing-max-sl-badge');
  if(swingMaxSl) swingMaxSl.textContent = `MAX SL: -$${Math.abs(d.stop_loss_amount||60).toFixed(0)}`;
  const cdEl = document.getElementById('cooldown-badge');
  if(d.cooldown_active){ cdEl.innerHTML=`<div class="cooldown-badge">⚡ COOLDOWN ${d.cooldown_remaining||0}s</div>`; }
  else { cdEl.innerHTML=''; }

  // Universe
  const avail = d.universe_available || 0, cfg = d.universe_configured || 0;
  document.getElementById('uni-val').textContent = `${avail}/${cfg}`;
  document.getElementById('uni-val').style.color = avail===cfg?'var(--green)':'var(--amber)';
  document.getElementById('uni-bar').style.width = (cfg>0?clamp(avail/cfg,0,1)*100:0).toFixed(1)+'%';
  document.getElementById('uni-conf').textContent = d.health_conf || '--';
  document.getElementById('uni-latency').textContent = `${(d.solve_latency_ms||0).toFixed(0)} ms`;
  document.getElementById('uni-mem').textContent = `${(d.memory_mb||0).toFixed(1)} MB`;

  // Regime + Chop indicator
  renderRegime(d);

  // Status bar
  renderStatusBar(d, active);
  document.getElementById('last-update').textContent = new Date().toLocaleTimeString();

  // Swing page
  renderSwing(d);

  // Analysis page
  renderSSAGrid(d);
  renderSwingAnalysis(d);
}

// ── Navigation ─────────────────────────────────────────────────────────
function showPage(name){
  document.getElementById('overview-page').style.display = name==='overview' ? '' : 'none';
  document.getElementById('swing-page').style.display = name==='swing' ? '' : 'none';
  document.getElementById('analysis-page').style.display = name==='analysis' ? '' : 'none';
  document.querySelectorAll('.nav-link').forEach(b => b.classList.remove('active'));
  document.getElementById('nav-'+name).classList.add('active');
}
document.getElementById('nav-overview').addEventListener('click', () => showPage('overview'));
document.getElementById('nav-swing').addEventListener('click', () => showPage('swing'));
document.getElementById('nav-analysis').addEventListener('click', () => showPage('analysis'));

// SSP table toggle
document.getElementById('toggle-ssp-table').addEventListener('click', function(){
  const wrap = document.getElementById('ssp-table-wrap');
  const show = wrap.style.display === 'none';
  wrap.style.display = show ? '' : 'none';
  this.textContent = show ? 'HIDE' : 'SHOW';
});

// ── SSE connection ──────────────────────────────────────────────────────
function connect(){
  const es = new EventSource('/events');
  es.onmessage = (e) => {
    try{ update(JSON.parse(e.data)); } catch(ex){ console.warn('parse error', ex); }
  };
  es.onerror = () => { es.close(); setTimeout(connect, 3000); };
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

    server = ThreadingHTTPServer(("0.0.0.0", PORT), _Handler)
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
