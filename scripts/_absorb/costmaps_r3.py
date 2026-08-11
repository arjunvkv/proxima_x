"""costmaps_r3.py — corrected tick/spread maps for engine runs on new assets."""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from proxima_ops.backtest.pnl import point_size as engine_pt  # noqa: E402
from proxima_ops.backtest.pnl import FTMO_TICK_VALUES as _ENGINE_TICK  # noqa: E402

_TV = json.load(open(os.path.join(ROOT, "scripts/_absorb/results/new_tick_values.json")))

FX_TV = dict(_ENGINE_TICK)  # all 22 pairs from the engine's canonical broker table
FX_TV.update({"CADJPY": 0.673, "AUDCAD": 0.716759, "AUDCHF": 1.234751, "EURCAD": 0.716759})  # 4 extras (class values)

# true measured live spreads in PIPS (pip = 10 x point)
FX_SPR = {"EURUSD": 0.6, "USDJPY": 0.6, "GBPUSD": 0.7, "AUDUSD": 0.6, "EURJPY": 0.7,
          "GBPJPY": 0.8, "EURAUD": 1.0, "EURNZD": 1.4, "GBPAUD": 1.1, "GBPNZD": 1.5,
          "GBPCAD": 1.0, "AUDNZD": 1.0, "USDCAD": 0.7, "NZDUSD": 0.7, "EURGBP": 0.6,
          "EURCHF": 0.9, "USDCHF": 0.8, "AUDJPY": 0.7,
          "CADJPY": 0.8, "AUDCAD": 0.8, "AUDCHF": 0.9, "EURCAD": 1.0}
NEW_SPR = {"XAUUSD": 4.5, "XAGUSD": 5.9, "BTCUSD": 10.0, "ETHUSD": 6.0,
           "US30.cash": 21.0, "US500.cash": 6.0, "GER40.cash": 11.3, "UK100.cash": 7.5,
           "JP225.cash": 100.0, "HK50.cash": 60.0, "USOIL.cash": 6.8, "UKOIL.cash": 6.7,
           "DXY.cash": 1.7}

ALL = list(FX_TV) + list(_TV.keys())

def corrected_maps():
    """Return (tick_map, spread_map) with engine point_size compensated."""
    tick, spr = dict(FX_TV), dict(FX_SPR)
    for sym in NEW_SPR:
        meta = _TV[sym]
        tv_true = meta["tv"]
        pt_true = meta["pt"]
        pe = engine_pt(sym)          # what the engine uses (0.00001 / 0.001)
        tick[sym] = tv_true * pe / pt_true
        spr[sym] = NEW_SPR[sym] * pt_true / pe
    return tick, spr

if __name__ == "__main__":
    t, s = corrected_maps()
    for sym in ["XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD", "US30.cash", "JP225.cash"]:
        print(f"{sym}: tv'={t[sym]:.6g} sp'={s[sym]:.6g}")
