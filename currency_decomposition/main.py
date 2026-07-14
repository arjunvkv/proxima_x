import sys
import os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

from runtime.manager import RuntimeManager

def main():
    runtime = RuntimeManager()
    runtime.start()

if __name__ == "__main__":
    main()
