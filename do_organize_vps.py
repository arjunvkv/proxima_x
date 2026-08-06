import subprocess

VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST = "ubuntu@140.245.234.92"
EXP_DIR = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Experts"

def ssh(cmd):
    full = f'ssh -i "{VPS_KEY}" {VPS_HOST} "{cmd}"'
    return subprocess.run(full, shell=True, capture_output=True, text=True).stdout

def main():
    print("=" * 100)
    print("PROXIMA X — ORGANIZING VPS MT5 EXPERTS BY VERSION SUBFOLDERS")
    print("=" * 100)

    ssh(f'mkdir -p "{EXP_DIR}/v8" "{EXP_DIR}/v107" "{EXP_DIR}/v106" "{EXP_DIR}/legacy"')
    ssh(f'mv "{EXP_DIR}"/*_v8.* "{EXP_DIR}/v8/" 2>/dev/null || true')
    ssh(f'mv "{EXP_DIR}"/*_v107.* "{EXP_DIR}/v107/" 2>/dev/null || true')
    ssh(f'mv "{EXP_DIR}"/*_v106.* "{EXP_DIR}/v106/" 2>/dev/null || true')
    ssh(f'mv "{EXP_DIR}"/*.mq5 "{EXP_DIR}/legacy/" 2>/dev/null || true')
    ssh(f'mv "{EXP_DIR}"/*.ex5 "{EXP_DIR}/legacy/" 2>/dev/null || true')

    print("\n📂 LIVE VPS MT5 EXPERTS DIRECTORY STRUCTURE:")
    print("=" * 100)

    for sub in ["v8", "v107", "v106", "legacy"]:
        res = ssh(f'ls -1 "{EXP_DIR}/{sub}"')
        items = [x.strip() for x in res.splitlines() if x.strip()]
        print(f"\n📂 Folder: Experts/{sub}/ ({len(items)} items):")
        for item in items:
            print(f"  • {item}")

    print("\n" + "=" * 100)
    print("🟢 COMPLETED: ALL VPS MT5 EXPERTS ARE ORGANIZED BY VERSION!")
    print("=" * 100)

if __name__ == "__main__":
    main()
