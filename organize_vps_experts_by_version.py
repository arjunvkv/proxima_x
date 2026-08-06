import subprocess

VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST = "ubuntu@140.245.234.92"
EXPERTS_DIR = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Experts"

def run_ssh(remote_cmd):
    full_cmd = f'ssh -i "{VPS_KEY}" {VPS_HOST} "{remote_cmd}"'
    res = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    return res.stdout, res.stderr

def main():
    print("=" * 115)
    print("PROXIMA X — ORGANIZING VPS MT5 EXPERTS FOLDER BY VERSION SUBFOLDERS")
    print("=" * 115)

    # 1. Check current contents of VPS Experts folder
    stdout, _ = run_ssh(f"ls -1 '{EXPERTS_DIR}'")
    files = [f.strip() for f in stdout.split("\n") if f.strip()]

    print(f"Total items in VPS Experts directory: {len(files)}")

    # Create subdirectories on VPS if they don't exist
    commands = [
        f"mkdir -p '{EXPERTS_DIR}/v8'",
        f"mkdir -p '{EXPERTS_DIR}/v107'",
        f"mkdir -p '{EXPERTS_DIR}/v106'",
        f"mkdir -p '{EXPERTS_DIR}/legacy'",
    ]

    for cmd in commands:
        run_ssh(cmd)

    # Move files into respective version folders
    # v8 files -> v8/
    run_ssh(f"mv '{EXPERTS_DIR}'/*_v8.* '{EXPERTS_DIR}/v8/' 2>/dev/null || true")
    
    # v107 files -> v107/
    run_ssh(f"mv '{EXPERTS_DIR}'/*_v107.* '{EXPERTS_DIR}/v107/' 2>/dev/null || true")

    # v106 files -> v106/
    run_ssh(f"mv '{EXPERTS_DIR}'/*_v106.* '{EXPERTS_DIR}/v106/' 2>/dev/null || true")

    # legacy files -> legacy/
    run_ssh(f"mv '{EXPERTS_DIR}'/*.mq5 '{EXPERTS_DIR}/legacy/' 2>/dev/null || true")
    run_ssh(f"mv '{EXPERTS_DIR}'/*.ex5 '{EXPERTS_DIR}/legacy/' 2>/dev/null || true")

    # Verify structure on VPS
    print("\n📁 UPDATED VPS MT5 EXPERTS DIRECTORY STRUCTURE:")
    print("=" * 115)

    subfolders = ["v8", "v107", "v106", "legacy"]
    for sub in subfolders:
        out, _ = run_ssh(f"ls -1 '{EXPERTS_DIR}/{sub}'")
        sub_files = [f.strip() for f in out.split("\n") if f.strip()]
        print(f"\n📂 Subfolder '{sub}/' ({len(sub_files)} items):")
        for sf in sub_files:
            print(f"  • {sf}")

    print("\n" + "=" * 115)
    print("🟢 VPS MT5 EXPERTS FOLDER ORGANIZED INTO VERSION SUBFOLDERS SUCCESS!")
    print("=" * 115)

if __name__ == "__main__":
    main()
