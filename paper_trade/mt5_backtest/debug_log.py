log_path = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\81A933A9AFC5DE3C23B15CAB19C63850\Tester\logs\20260727.log"

# Read as raw bytes, check encoding
with open(log_path, 'rb') as f:
    raw = f.read(10000)

print("First 200 bytes:")
print(raw[:200])
print()

# Count lines
lines = raw.decode('utf-8', errors='replace').split('\n')
print(f"First 5 lines:")
for i, l in enumerate(lines[:5]):
    print(f"  [{i}] {repr(l[:120])}")

print()
# Search for trade patterns
for i, line in enumerate(lines):
    # Check without encoding
    raw_line = raw.split(b'\n')[i] if i < len(raw.split(b'\n')) else b''
    if b'OPEN' in raw_line and b'dir=' in raw_line:
        print(f"Found OPEN at line {i}: {raw_line[:200]}")
        break
else:
    print("No OPEN dir= found in first 10000 bytes")
    print("Sample raw lines with OPEN:")
    for i, raw_line in enumerate(raw.split(b'\n')[:50]):
        if b'OPEN' in raw_line or b'CLOSE' in raw_line or b'LOST' in raw_line:
            print(f"  [{i}] {raw_line[:200]}")
