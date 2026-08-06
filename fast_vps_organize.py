import subprocess

VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST = "ubuntu@140.245.234.92"
EXP_DIR = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Experts"

def main():
    print("=" * 100)
    print("PROXIMA X — ORGANIZING VPS EXPERTS DIRECTORY BY VERSION")
    print("=" * 100)

    remote_bash = (
        f"cd '{EXP_DIR}' && "
        "mkdir -p v8 v107 v106 legacy && "
        "mv *_v8.* v8/ 2>/dev/null || true; "
        "mv *_v107.* v107/ 2>/dev/null || true; "
        "mv *_v106.* v106/ 2>/dev/null || true; "
        "mv *.mq5 legacy/ 2>/dev/null || true; "
        "mv *.ex5 legacy/ 2>/dev/null || true; "
        "echo '=== v8 ==='; ls -1 v8/; "
        "echo '=== v107 ==='; ls -1 v107/; "
        "echo '=== v106 ==='; ls -1 v106/; "
        "echo '=== legacy ==='; ls -1 legacy/"
    )

    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, VPS_HOST, remote_bash]
    res = subprocess.run(cmd, capture_output=True, text=True)

    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)
    print("=" * 100)
    print("🟢 COMPLETED!")

if __name__ == "__main__":
    main()
