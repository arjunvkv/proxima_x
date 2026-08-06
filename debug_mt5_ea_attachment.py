import subprocess

VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST = "ubuntu@140.245.234.92"
MT5_ROOT = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal"

def ssh(cmd):
    full = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, VPS_HOST, f"bash -c '{cmd}'"]
    return subprocess.run(full, capture_output=True, text=True).stdout

def main():
    print("=" * 100)
    print("PROXIMA X — DIAGNOSING MT5 EA ATTACHMENT STATUS ON VPS...")
    print("=" * 100)

    # 1. Check where EX5 files exist
    print("📂 Checking EX5 File Paths in Experts:")
    out1 = ssh(f"ls -la '{MT5_ROOT}/MQL5/Experts/' '{MT5_ROOT}/MQL5/Experts/v8/' 2>/dev/null")
    print(out1)

    # 2. Check MT5 Logs for EA attachment errors
    print("\n📋 Checking Recent MT5 Log Messages:")
    out2 = ssh(f"tail -n 30 '{MT5_ROOT}/MQL5/Logs/'*.log 2>/dev/null || tail -n 30 '{MT5_ROOT}/Logs/'*.log 2>/dev/null")
    print(out2)

    print("=" * 100)

if __name__ == "__main__":
    main()
