import subprocess

VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST = "ubuntu@140.245.234.92"
CHARTS_DIR = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/Profiles/Charts/Default"

def main():
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, VPS_HOST, f"ls -la '{CHARTS_DIR}'"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("Default profile contents:")
    print(res.stdout)

    # Cat first chart file to see structure
    cmd2 = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, VPS_HOST, f"head -n 40 '{CHARTS_DIR}/chart01.chr'"]
    res2 = subprocess.run(cmd2, capture_output=True, text=True)
    print("\nSample chart01.chr structure:")
    print(res2.stdout)

if __name__ == "__main__":
    main()
