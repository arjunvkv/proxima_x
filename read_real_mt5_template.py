import subprocess

VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST = "ubuntu@140.245.234.92"
TPL_DIR = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Profiles/Templates"

def main():
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, VPS_HOST, f"python3 -c \"import sys; print(open('{TPL_DIR}/ADX.tpl', 'rb').read().decode('utf-16le'))\""]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("Real MT5 ADX.tpl template content:")
    print(res.stdout[:2000])

if __name__ == "__main__":
    main()
