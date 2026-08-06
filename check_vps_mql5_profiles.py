import subprocess

VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST = "ubuntu@140.245.234.92"
MQL5_CHARTS = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Profiles/Charts"

def main():
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, VPS_HOST, f"ls -la '{MQL5_CHARTS}'"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("MQL5/Profiles/Charts directory contents:")
    print(res.stdout)

if __name__ == "__main__":
    main()
