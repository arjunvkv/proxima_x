# Debug the CSV file
import csv

csv_path = "C:\\Trading\\Agentic_Trading\\proxima_x\\paper_trade\\mt5_backtest\\custom_data\\FN_EURUSD.csv"
with open(csv_path, "rb") as f:
    raw = f.read(200)

print("First 100 bytes:")
print(repr(raw[:100]))
print("Last 100 bytes:")
print(repr(raw[-100:]))

lines_b = raw.split(b"\n")
line0_str = lines_b[0].decode("utf-8")
print(f"First line: [{line0_str}]")

# Check BOM
if raw[:3] == b"\xef\xbb\xbf":
    print("HAS UTF-8 BOM")
else:
    print(f"No BOM. First 4 hex: {raw[:4].hex()}")

# Check line endings
if b"\r\n" in raw[:100]:
    print("Line endings: CRLF")
elif b"\n" in raw[:100]:
    print("Line endings: LF")

# Count lines
with open(csv_path) as f:
    lines = f.readlines()
print(f"Total lines: {len(lines)}")

# Parse CSV properly
with open(csv_path, newline="") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i < 3:
            print(f"Row {i}: {row} (len={len(row)})")
    # Count rows
    total = sum(1 for _ in reader) + 3
    print(f"Total CSV rows: {total}")
