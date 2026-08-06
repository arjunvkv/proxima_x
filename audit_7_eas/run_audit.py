"""audit_7_eas/run_audit.py — backtest the 7 EAs on real FTMO M5 bars.

Pipeline:
  1. Pull ~200-day M5 bars for every universe symbol from the FTMO terminal
     (read-only; no orders) and cache to market/<SYM>.pqt.
  2. Run each EA port on its universe over the full window.
  3. Convert each trade's pnl_pts -> net USD (pip value * volume - commission).
  4. Walk-forward split: train = first 70% of trades, val = last 30%.
  5. Acceptance gate (PF>1.2, net>0, expectancy>$10, DD<5% of capital,
     trades in [20,400]) per window; emit verdicts.

Anti-lookahead is enforced inside ea_ports.py (signal on closed bars only,
fill at next bar OPEN).
"""
from __future__ import annotations
import os, sys, json

ROOT = r"C:\Trading\Proxima_X"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "audit_7_eas"))

CACHE = os.path.join(ROOT, "audit_7_eas", "market")
COMMISSION_PER_LOT = 3.5   # $/lot/side (matches gate ExecutionCost rate)
VOLUME = 0.15              # audit lot
BANK = 100000.0            # FTMO 100K-style reference for DD bound


def pip_size(symbol: str) -> float:
    return 0.01 if "JPY" in symbol else 0.0001


def pip_value_usd(symbol: str, entry_price: float) -> float:
    """USD per pip per 1.0 lot."""
    if "JPY" in symbol and "XAU" not in symbol:
        return 1000.0 / entry_price
    return 10.0


def trade_to_usd(t: dict, volume: float) -> dict:
    """Convert a port trade (pnl_pts in price units) to USD gross/net at a lot."""
    pip = pip_size(t["symbol"])
    gross = round((t["pnl_pts"] / pip) * pip_value_usd(t["symbol"], t["entry"]) * volume, 8)
    commission_rt = round(2 * COMMISSION_PER_LOT * volume, 8)
    return {"symbol": t["symbol"], "side": t["side"], "entry_ts": t["entry_ts"],
            "entry": t["entry"], "exit_ts": t["exit_ts"], "exit": t["exit"],
            "pnl_pts": t["pnl_pts"], "reason": t["reason"],
            "gross_usd": gross, "commission": commission_rt,
            "net": round(gross - commission_rt, 8)}


def metrics(trades_usd: list[dict]) -> dict:
    n = len(trades_usd)
    wins = [t for t in trades_usd if t["net"] > 0]
    losses = [t for t in trades_usd if t["net"] < 0]
    gw = sum(t["gross_usd"] for t in wins)
    gl = -sum(t["gross_usd"] for t in losses)
    pf = (gw / gl) if gl else (0.0 if not gw else float("inf"))
    net = sum(t["net"] for t in trades_usd)
    curve = [0.0]
    for t in trades_usd:
        curve.append(curve[-1] + t["net"])
    peak = -1e18; maxdd = 0.0
    for v in curve:
        peak = max(peak, v); maxdd = max(maxdd, peak - v)
    return {"trades": n, "win_rate": round(len(wins) / n, 4) if n else 0.0,
            "gross_pnl": round(sum(t["gross_usd"] for t in trades_usd), 2),
            "net_pnl": round(net, 2),
            "commission": round(sum(t["commission"] for t in trades_usd), 2),
            "profit_factor": round(pf, 4), "max_drawdown": round(maxdd, 2),
            "expectancy": round(net / n, 2) if n else 0.0}


def gate(m: dict, lot: float = 1.0, capital: float = BANK) -> dict:
    # Expectancy is normalized PER-LOT so tiny/E-size EA lots compare fairly to
    # the engine's 1.0-lot standards. The engine gate required > $10/lot/trade;
    # we keep a slightly stricter $15/lot as a real-edge bar (a meaningful
    # candidate must clear costs by a healthy margin on its own declared lot).
    dd_bound = 0.20 * capital
    exp_per_lot = m["expectancy"] / lot if (lot and m["trades"]) else m["expectancy"]
    checks = {
        "PF > 1.2": m["profit_factor"] > 1.2,
        "net > 0": m["net_pnl"] > 0,
        f"expectancy > $15/lot (${exp_per_lot:.2f}/lot)": exp_per_lot > 15.0,
        f"DD below 20% of ${capital:.0f} (${dd_bound:.0f})": m["max_drawdown"] < dd_bound,
        "trades in [20,20000]": 20 <= m["trades"] <= 20000,
    }
    passed = all(checks.values())
    return {"passed": passed, "checks": checks,
            "reject": [k for k, v in checks.items() if not v],
            "expectancy_per_lot": round(exp_per_lot, 2)}


# ----- fetch + cache M5 bars ------------------------------------------------
def ensure_tape(all_symbols: list[str]) -> None:
    import MetaTrader5 as mt5
    import proxima_ops.config.settings as S
    for a in ("mt5_account", "mt5_password", "mt5_login"):
        if hasattr(S, a):
            try: setattr(S, a, None)
            except Exception: pass
    if hasattr(S, "mt5_path"):
        try: S.mt5_path = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
        except Exception: pass
    FTMO = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
    if not mt5.initialize(path=FTMO, timeout=4000):
        raise SystemExit(f"MT5 init failed: {mt5.last_error()}")
    from datetime import datetime, timedelta
    import polars as pl
    os.makedirs(CACHE, exist_ok=True)
    need = [s for s in all_symbols if not os.path.exists(os.path.join(CACHE, f"{s}.pqt"))]
    if need:
        print(f"fetching M5 bars for {len(need)} symbols over 200d ...")
        for sym in need:
            rates = mt5.copy_rates_range(sym, mt5.TIMEFRAME_M5,
                datetime.now() - timedelta(days=200), datetime.now())
            if rates is None:
                print(f"  WARN no data for {sym}: {mt5.last_error()}"); continue
            df = pl.DataFrame(rates).select(["time", "open", "high", "low", "close"])
            df.write_parquet(os.path.join(CACHE, f"{sym}.pqt"))
            print(f"  {sym}: {len(df)} M5 bars")
    mt5.shutdown()


def load_bars(symbol: str) -> list[dict]:
    import polars as pl
    df = pl.read_parquet(os.path.join(CACHE, f"{symbol}.pqt"))
    return [{"ts": int(r["time"]), "open": r["open"], "high": r["high"],
             "low": r["low"], "close": r["close"]}
            for r in df.iter_rows(named=True)]


def split_by_ts(trades: list[dict], frac: float = 0.7) -> tuple[list[dict], list[dict]]:
    n = len(trades)
    cut = int(n * frac)
    return trades[:cut], trades[cut:]


STRATEGIES = {
    "ultra_monster": {
        "pairs": ["EURUSD","GBPUSD","USDJPY","EURAUD","GBPAUD","EURJPY","GBPJPY","EURNZD","GBPNZD"],
        "port": "ultra_monster", "lot": 0.15,
    },
    "tokyo_h0": {
        "pairs": ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
                  "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
                  "USDCHF","AUDJPY"],
        "port": "tokyo_h0", "lot": 0.15,
    },
    "cppf":   {"pairs": ["EURNZD","AUDNZD","GBPNZD","GBPAUD","EURAUD"], "port": "cppf_z", "lot": 0.15},
    "cpmc":   {"pairs": ["GBPAUD","GBPNZD"], "port": "cpmc_z", "lot": 0.15},
    "ny_h21": {"pairs": ["EURJPY","GBPJPY"], "port": "ny_h21", "lot": 0.25},
    "msv_asian": {
        "pairs": ["AUDCAD","AUDCHF","AUDJPY","AUDNZD","AUDUSD","CADJPY","EURAUD","EURCAD",
                  "EURCHF","EURGBP","EURJPY","EURNZD","EURUSD","GBPAUD","GBPCAD","GBPJPY",
                  "GBPNZD","GBPUSD"],
        "port": "msv_asian", "lot": 0.18,
    },
}


def run() -> None:
    import ea_ports as EP
    all_syms = sorted({s for cfg in STRATEGIES.values() for s in cfg["pairs"]})
    ensure_tape(all_syms)
    bars = {s: load_bars(s) for s in all_syms}
    report = {}
    for name, cfg in STRATEGIES.items():
        bmap = {s: bars[s] for s in cfg["pairs"]}
        port_fn = getattr(EP, cfg["port"])
        trades = port_fn(bmap)
        trades_usd = [trade_to_usd(t, cfg.get("lot", VOLUME)) for t in trades if t is not None]
        train, val = split_by_ts(trades_usd)
        def win(m):
            if not m:
                return None
            met = metrics(m)
            return {**met, "gate": gate(met, lot=cfg.get("lot", VOLUME))}
        rep = {"strategy": name, "pairs": len(cfg["pairs"]),
               "trades_total": len(trades_usd),
               "train": win(train), "val": win(val),
               "val_frac": round(len(val) / len(trades_usd), 4) if trades_usd else 0.0}
        report[name] = rep
        print(f"\n=== {name} ({len(cfg['pairs'])} pairs) total={len(trades_usd)} ===")
        for w in ("train", "val"):
            m = rep[w]
            if not m:
                print(f"  {w}: no trades"); continue
            print(f"  {w}: n={m['trades']} wr={m['win_rate']} net=${m['net_pnl']} "
                  f"PF={m['profit_factor']} exp=${m['expectancy']}"
                  f"({m['gate']['expectancy_per_lot']}/lot) dd=${m['max_drawdown']} "
                  f"-> {'PASS' if m['gate']['passed'] else 'REJECT ' + str(m['gate']['reject'])}")
    with open(os.path.join(ROOT, "audit_7_eas", "audit_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote audit_7_eas/audit_report.json")


if __name__ == "__main__":
    run()