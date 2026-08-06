import subprocess

VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST = "ubuntu@140.245.234.92"
MQL5_LOG = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/logs/20260804.log"
TERM_LOG = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/logs/20260804.log"

def main():
    print("=" * 100)
    print("READING VPS MT5 TERMINAL & MQL5 LOGS FOR TODAY (2026-08-04)...")
    print("=" * 100)

    cmd1 = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, VPS_HOST, f"tail -n 50 '{MQL5_LOG}'"]
    res1 = subprocess.run(cmd1, capture_output=True, text=True)
    print("MQL5 Log (last 50 lines):")
    print(res1.stdout)

    print("\n" + "=" * 100)
    cmd2 = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, VPS_HOST, f"tail -n 50 '{TERM_LOG}'"]
    res2 = subprocess.run(cmd2, capture_output=True, text=True)
    print("Terminal Log (last 50 lines):")
    print(res2.stdout)

if __name__ == "__main__":
    main()
