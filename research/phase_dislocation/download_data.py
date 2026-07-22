"""Batch download Dukascopy M1 data for Currency Inventory Pressure backtest."""
import subprocess, os, glob, shutil, pandas as pd

outdir = 'research/phase_dislocation/dukascopy_data'
download_dir = os.path.join(outdir, 'download')
os.makedirs(download_dir, exist_ok=True)

# Pairs needed for currency indices (all 28)
# Already have: EURUSD, USDJPY, GBPUSD, AUDUSD, NZDUSD, EURJPY, GBPJPY (in temp data)
# Need to download: the rest

needed_pairs = [
    'eurnzd', 'gbpnzd', 'audnzd', 'nzdjpy', 'nzdcad', 'nzdchf',  # NZD crosses
    'eurchf', 'gbpchf', 'audchf', 'cadchf', 'chfjpy',             # CHF crosses  
    'eurcadjpy',  # oops
]

# Be precise: list all pairs needed for NZD, CHF, USD, GBP indices
pairs_to_download = [
    'eurnzd', 'gbpnzd', 'audnzd', 'nzdjpy', 'nzdcad', 'nzdchf',
    'eurchf', 'gbpchf', 'audchf', 'cadchf', 'chfjpy',
    'eurgbp', 'euraud', 'eurcad',
    'gbpaud', 'gbpcad', 'gbpnzd',
    'audcad', 'audnzd', 'audchf',
    'usdcad',
    'nzdusd',  # 1 day only in market, need more
    'usdchf',  # 1 day only
]

# Remove duplicates
pairs_to_download = list(dict.fromkeys(pairs_to_download))
print(f"Pairs to download: {len(pairs_to_download)}")
for p in pairs_to_download:
    print(f"  {p}")

# Download 3 months: Apr, May, Jun 2026
months = [
    ('2026-04-01', '2026-04-30'),
    ('2026-05-01', '2026-05-31'),
    ('2026-06-01', '2026-06-30'),
]

total = len(pairs_to_download) * len(months)
count = 0

for pair in pairs_to_download:
    for from_d, to_d in months:
        fname = f'{pair}-m1-bid-{from_d}-{to_d}.csv'
        fpath = os.path.join(download_dir, fname)
        if os.path.exists(fpath):
            print(f"  [{count+1}/{total}] Skipping {fname} (exists)")
            count += 1
            continue
        
        print(f"  [{count+1}/{total}] Downloading {pair} {from_d} -> {to_d}...")
        cmd = f'npx dukascopy-node -i {pair} -from {from_d} -to {to_d} -t m1 -f csv'
        res = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=outdir, shell=True
        )
        if res.returncode != 0:
            print(f"    ERROR: {res.stderr[:200]}")
        else:
            print(f"    OK ({len(res.stdout)} chars)")
        count += 1

# Convert all CSVs to a single parquet per pair
print("\nConverting CSVs to parquet...")
pair_files = {}
for f in sorted(glob.glob(os.path.join(download_dir, '*.csv'))):
    parts = os.path.basename(f).split('-')
    pair = parts[0]
    if pair not in pair_files:
        pair_files[pair] = []
    pair_files[pair].append(f)

for pair, files in pair_files.items():
    print(f"  {pair}: {len(files)} files")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        dfs.append(df)
    combined = pd.concat(dfs).sort_values('timestamp').drop_duplicates(subset='timestamp')
    combined.to_parquet(os.path.join(outdir, f'{pair}.parquet'))
    print(f"    -> {len(combined)} rows")

print("\nDone!")
