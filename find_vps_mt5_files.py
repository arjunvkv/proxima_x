import subprocess

VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST = "ubuntu@140.245.234.92"

def main():
    print("Finding all .ex5 files on VPS:")
    cmd1 = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, VPS_HOST, 'find /home/ubuntu/.wine/ -name "*.ex5" 2>/dev/null']
    res1 = subprocess.run(cmd1, capture_output=True, text=True)
    print(res1.stdout)

    print("\nFinding all .log files on VPS:")
    cmd2 = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, VPS_HOST, 'find /home/ubuntu/.wine/ -name "*.log" 2>/dev/null']
    res2 = subprocess.run(cmd2, capture_output=True, text=True)
    print(res2.stdout[:2000])

if __name__ == "__main__":
    main()
