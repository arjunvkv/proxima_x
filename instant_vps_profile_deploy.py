import os
import subprocess
import tarfile

VPS_KEY = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"
VPS_HOST = "ubuntu@140.245.234.92"
MQL5_CHARTS = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Profiles/Charts"
ROOT_CHARTS = "/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/Profiles/Charts"
LOCAL_TEMP_DIR = r"C:\Trading\Agentic_Trading\proxima_x\temp_profile_v8"
TAR_PATH = r"C:\Trading\Agentic_Trading\proxima_x\v8_charts.tar"

def main():
    print("=" * 100)
    print("INSTANT PROFILE DEPLOYMENT TO VPS MQL5/Profiles/Charts...")
    print("=" * 100)

    # 1. Create tar file
    with tarfile.open(TAR_PATH, "w") as tar:
        for f in os.listdir(LOCAL_TEMP_DIR):
            full_p = os.path.join(LOCAL_TEMP_DIR, f)
            tar.add(full_p, arcname=f)

    print(f"Packed {len(os.listdir(LOCAL_TEMP_DIR))} chart files into v8_charts.tar ({os.path.getsize(TAR_PATH):,} bytes)")

    # 2. SCP single tar file
    print("Uploading tar archive to VPS...")
    subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, TAR_PATH, f"{VPS_HOST}:/tmp/v8_charts.tar"], check=True)

    # 3. Untar on VPS into all profile destinations
    remote_script = (
        f"mkdir -p '{MQL5_CHARTS}/Proxima_v8' '{MQL5_CHARTS}/Default' '{ROOT_CHARTS}/Proxima_v8' '{ROOT_CHARTS}/Default' && "
        f"tar -xf /tmp/v8_charts.tar -C '{MQL5_CHARTS}/Proxima_v8/' && "
        f"tar -xf /tmp/v8_charts.tar -C '{MQL5_CHARTS}/Default/' && "
        f"tar -xf /tmp/v8_charts.tar -C '{ROOT_CHARTS}/Proxima_v8/' && "
        f"tar -xf /tmp/v8_charts.tar -C '{ROOT_CHARTS}/Default/' && "
        f"rm /tmp/v8_charts.tar && "
        f"echo '=== MQL5 Profiles/Charts ===' && ls -la '{MQL5_CHARTS}'"
    )

    print("Extracting on VPS...")
    res = subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-i", VPS_KEY, VPS_HOST, f"bash -c \"{remote_script}\""], capture_output=True, text=True)

    print(res.stdout)
    print("=" * 100)
    print("🟢 COMPLETED INSTANTLY!")

if __name__ == "__main__":
    main()
