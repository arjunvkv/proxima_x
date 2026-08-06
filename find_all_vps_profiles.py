import subprocess

VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST = "ubuntu@140.245.234.92"

def main():
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, VPS_HOST, 'find /home/ubuntu/.wine/ -iname "*Charts*" 2>/dev/null']
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("Found Charts paths on VPS:")
    print(res.stdout)

if __name__ == "__main__":
    main()
