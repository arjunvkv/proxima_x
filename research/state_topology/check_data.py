"""Check what data is available for STL state space construction."""
import numpy as np

d = np.load("C:/Trading/Agentic_Trading/proxima_x/research/directional_state/cache/EURJPY_state.npz")
print("Available arrays:", list(d.keys()))
for k in d.keys():
    arr = d[k]
    nnan = np.isnan(arr).sum() if arr.dtype.kind != "i" else "N/A"
    u = np.unique(arr[~np.isnan(arr)])[:10] if arr.dtype.kind != "i" else np.unique(arr)[:10]
    print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}, nan={nnan}, unique_preview={u}")
