import subprocess

VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST = "ubuntu@140.245.234.92"
TPL_DIR = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Profiles/Templates"

def main():
    print("=" * 100)
    print("INSPECTING MT5 TEMPLATES ON VPS...")
    print("=" * 100)

    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, VPS_HOST, f"ls -la '{TPL_DIR}'"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)

    cmd2 = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, VPS_HOST, f"cat '{TPL_DIR}/default.tpl' 2>/dev/null || cat '{TPL_DIR}/Default.tpl' 2>/dev/null"]
    res2 = subprocess.run(cmd2, capture_output=True, text=True)
    print("\nDefault.tpl content preview:")
    print(res2.stdout[:1500])

if __name__ == "__main__":
    main()
