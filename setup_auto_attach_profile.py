import subprocess

VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST = "ubuntu@140.245.234.92"
MT5_DIR = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal"
PROFILES_DIR = f"{MT5_DIR}/MQL5/Profiles"

def ssh(cmd):
    full = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, VPS_HOST, f"bash -c '{cmd}'"]
    return subprocess.run(full, capture_output=True, text=True).stdout

def main():
    print("=" * 100)
    print("PROXIMA X — AUTOMATED MT5 CHART & EA AUTO-ATTACHMENT SYSTEM")
    print("=" * 100)

    # Check Profiles directory
    out = ssh(f"ls -la '{MT5_DIR}/Profiles/Charts' 2>/dev/null || ls -la '{PROFILES_DIR}'")
    print("Existing MT5 Profiles:")
    print(out)

    print("=" * 100)

if __name__ == "__main__":
    main()
