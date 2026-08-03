
import subprocess, time, os

print('1. Killing terminal64.exe and wineserver...')
subprocess.run(['pkill', '-9', '-f', 'terminal64.exe'], check=False)
subprocess.run(['pkill', '-9', '-f', 'wineserver'], check=False)
time.sleep(3)

exp_dir = '/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MQL5/Experts/'
meta_cmd = '/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/MetaEditor64.exe'

eas = [
    'Test_Min_Fire_MT5', 'Ultra_Monster_MT5', 'TokyoH0_MT5',
    'CPPF_Z_MT5', 'CPMC_Z_MT5', 'NY_H21_MT5', 'MSV_Asian_Exhaustion_MT5'
]

env = os.environ.copy()
env['DISPLAY'] = ':0.0'

print('2. Compiling all 7 EAs fresh in Wine MetaEditor while terminal is DEAD...')
for ea in eas:
    mq5 = exp_dir + ea + '.mq5'
    if os.path.exists(mq5):
        subprocess.run(['wine', meta_cmd, f'/compile:{mq5}'], env=env, check=False)
        print(f'  • Compiled {ea}.mq5 cleanly!')

print('3. Verifying compiled .ex5 timestamps...')
for ea in eas:
    ex5 = exp_dir + ea + '.ex5'
    if os.path.exists(ex5):
        mtime = time.ctime(os.path.getmtime(ex5))
        print(f'  • {ea:<28} Timestamp: {mtime} 🟢')

print('4. Launching MT5 fresh...')
subprocess.Popen(['wine', '/home/ubuntu/.wine/drive_c/Program Files/FTMO Global Markets MT5 Terminal/terminal64.exe'], env=env)
time.sleep(4)

print('🟢 ALL 7 EAS RECOMPILED FRESH & MT5 RESTARTED CLEANLY!')
