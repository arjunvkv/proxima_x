"""Global config registry. Each strategy registers its config here."""
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STRATEGIES = {}  # name -> {pairs, pairs_file, hold_bars, session, mag_thresh, ...}

def register(name, cfg):
    STRATEGIES[name] = cfg

def get(name):
    return STRATEGIES.get(name)

def list_strategies():
    return list(STRATEGIES.keys())

LIVE_DIR = os.path.join(BASE, "live")
LOGS_DIR = os.path.join(LIVE_DIR, "logs")
REPORTS_DIR = os.path.join(LIVE_DIR, "reports")
