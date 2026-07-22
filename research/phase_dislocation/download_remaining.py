"""Download remaining 6 pairs to complete 28-pair universe."""
import subprocess, os, glob, pandas as pd

outdir = 'research/phase_dislocation/dukascopy_data'
download_dir = os.path.join(outdir, 'download')
os.makedirs(download_dir, exist_ok=True)

pairs = ['eurusd', 'usdjpy', 'gbpusd', 'audusd', 'eurjpy', 'gbpjpy']
months = [('2026-04-01', '2026-04-30'), ('2026-05-01', '2026-05-31'), ('2026-06-01', '2026-06-30')]

total = len(pairs) * len(months)
count = 0
for pair in pairs:
    for from_d, to_d in months:
        fname = f'{pair}-m1-bid-{from_d}-{to_d}.csv'
        count += 1
        if os.path.exists(os.path.join(download_dir, fname)):
            print(f"[{count}/{total}] Skipping {fname}")
            continue
        print(f"[{count}/{total}] Downloading {pair} {from_d} -> {to_d}...")
        cmd = f'npx dukascopy-node -i {pair} -from {from_d} -to {to_d} -t m1 -f csv'
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=outdir, shell=True)
        if res.returncode != 0:
            print(f"  ERROR: {res.stderr[:200]}")
        else:
            print(f"  OK")

# Convert to parquet
for pair in pairs:
    files = sorted(glob.glob(os.path.join(download_dir, f'{pair}-m1-bid-*.csv')))
    if not files:
        continue
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        dfs.append(df)
    combined = pd.concat(dfs).sort_values('timestamp').drop_duplicates(subset='timestamp')
    combined.to_parquet(os.path.join(outdir, f'{pair}.parquet'))
    print(f"  {pair}: {len(combined)} rows")
