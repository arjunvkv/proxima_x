import sys
import os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.manager import RuntimeManager

def main():
    runtime = RuntimeManager()
    runtime.start()

if __name__ == "__main__":
    main()
