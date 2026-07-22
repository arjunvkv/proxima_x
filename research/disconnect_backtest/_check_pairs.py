import glob, os
DATA_DIR = "research/phase_dislocation/dukascopy_data"
pairs = sorted([os.path.basename(f).replace(".parquet","").upper() for f in glob.glob(os.path.join(DATA_DIR,"*.parquet"))])
print("Parquet pairs (%d): %s" % (len(pairs), pairs))

BEST_PAIR = {"USD":"AUDUSD","EUR":"EURUSD","JPY":"NZDJPY","GBP":"GBPUSD","AUD":"AUDUSD","NZD":"NZDUSD","CAD":"NZDCAD","CHF":"USDCHF"}
print("\nBEST_PAIR targets:")
for c, bp in BEST_PAIR.items():
    status = "MATCH" if bp in pairs else "MISSING"
    print("  %s -> %s: %s" % (c, bp, status))
