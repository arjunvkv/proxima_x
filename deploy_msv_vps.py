#!/usr/bin/env python3
"""Deploy MSV_Asian_Exhaustion_MT5 to Oracle Cloud VPS."""
import paramiko, os

vps_ip = "140.245.234.92"
vps_user = "ubuntu"
key_path = r"C:\Users\Arjun Sasi\Downloads\ssh-key-2026-07-28.key"

local_mq5 = r"c:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\MSV_Asian_Exhaustion_MT5.mq5"
local_ex5 = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\89FE26BBBAB28C077BBF5FA8C1B4DF1C\MQL5\Experts\MSV_Asian_Exhaustion_MT5.ex5"

remote_dir = "/home/ubuntu/.wine_fundednext/drive_c/Program Files/FundedNext MT5 Terminal/MQL5/Experts/"

print(f"Connecting to VPS {vps_ip}...")
key = paramiko.RSAKey.from_private_key_file(key_path)
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(vps_ip, username=vps_user, pkey=key)

sftp = ssh.open_sftp()
print("Uploading MSV_Asian_Exhaustion_MT5.mq5...")
sftp.put(local_mq5, remote_dir + "MSV_Asian_Exhaustion_MT5.mq5")

print("Uploading MSV_Asian_Exhaustion_MT5.ex5...")
sftp.put(local_ex5, remote_dir + "MSV_Asian_Exhaustion_MT5.ex5")

sftp.close()

# Verify on VPS
stdin, stdout, stderr = ssh.exec_command(f"ls -l '{remote_dir}MSV_Asian_Exhaustion_MT5.*'")
print("\nVPS Verification:")
print(stdout.read().decode())

ssh.close()
print("Successfully deployed Strategy #5 (MSV Asian Exhaustion) to VPS!")
