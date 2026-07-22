from paper_trade.core.config import CONFIG, STRATEGIES
from paper_trade.strategies.currency_pressure.strategy import STRATEGY_NAME
cfg = STRATEGIES[STRATEGY_NAME]
print(f"mt5_account: {cfg.get('mt5_account')}")
print(f"mt5_path: {cfg.get('mt5_path')}")
print(f"pairs count: {len(cfg.get('pairs', []))}")
