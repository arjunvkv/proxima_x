import polars as pl
import numpy as np
from datetime import datetime
from collections import defaultdict

DATA_BASE = "C:/Trading/Agentic_Trading/data/intraday"
SYMBOLS = ["XAUUSD", "EURJPY", "USDJPY", "GBPJPY"]

def load_m5(symbol):
    df = pl.read_parquet(f"{DATA_BASE}/{symbol}_M5.parquet")
    arr = df.to_numpy()
    c = arr[:, 4].astype(np.float64)
    ts = arr[:, 0].astype(np.int64)
    return {"close": c, "ts": ts, "n": len(c)}

for sym in SYMBOLS:
    data = load_m5(sym)
    close = data["close"]
    ts = data["ts"]
    
    rets = np.diff(np.log(close), prepend=np.log(close[0]))
    
    years = np.array([datetime.fromtimestamp(t).year for t in ts])
    uniq_years = sorted(set(years))
    
    print(f"\n=== {sym} ===")
    print(f"Years: {uniq_years}")
    
    tr = np.abs(np.diff(close, prepend=close[0]))
    
    for yr in uniq_years:
        mask = years == yr
        yr_close = close[mask]
        yr_rets = np.diff(np.log(yr_close), prepend=np.log(yr_close[0]))
        yr_tr = np.abs(np.diff(yr_close, prepend=yr_close[0]))
        
        p_up = np.mean(yr_rets > 0)
        p_down = np.mean(yr_rets < 0)
        
        tr_p95 = np.percentile(yr_tr, 95)
        impulse_mask = yr_tr > tr_p95
        
        p_cont_given_impulse = np.mean((yr_rets[1:] > 0)[impulse_mask[1:]]) if np.sum(impulse_mask[1:]) > 0 else 0
        p_cont_unconditional = np.mean(yr_rets[1:] > 0)
        
        n_imp = int(np.sum(impulse_mask[1:]))
        
        print(f"  {yr}: P(up)={p_up:.4f}, P(up|impulse)={p_cont_given_impulse:.4f}, unconditional={p_cont_unconditional:.4f}, n_imp={n_imp}, imp_rate={n_imp/len(yr_close)*100:.1f}%")
