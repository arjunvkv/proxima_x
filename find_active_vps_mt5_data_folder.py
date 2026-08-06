import subprocess

VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST = "ubuntu@140.245.234.92"

def main():
    print("=" * 100)
    print("PROXIMA X — SEARCHING ALL MQL5/Experts DIRECTORIES ON VPS...")
    print("=" * 100)

    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, VPS_HOST, 'find /home/ubuntu/.wine/ -iname "Experts" -type d 2>/dev/null']
    res = subprocess.run(cmd, capture_output=True, text=True)

    paths = [p.strip() for p in res.stdout.splitlines() if p.strip()]
    print(f"Found {len(paths)} Experts directories on VPS:")
    for p in paths:
        print(f"  • {p}")

    print("=" * 100)

if __name__ == "__main__":
    main()
