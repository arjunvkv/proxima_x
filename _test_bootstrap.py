"""Quick diagnostic for OSS bootstrap."""
import sys, os, logging
sys.path.insert(0, os.getcwd())
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

from bootstrap.oss_bootstrap_trainer import OSSBootstrapTrainer
from bootstrap.market_seed import MarketSeedLoader

loader = MarketSeedLoader()
bootstrap = loader.seed_all()
print(f"Bootstrap symbols: {len(bootstrap)}", flush=True)

trainer = OSSBootstrapTrainer()
result = trainer.train_all(bootstrap)
print(f"Trained: {result['trained']}", flush=True)
print(f"Trained syms: {sum(1 for s,r in result.get('symbols',{}).items() if r.get('trained'))}/{len(result.get('symbols',{}))}", flush=True)

# Check _trained flag
print(f"trainer._trained = {trainer._trained}", flush=True)

# Verify our code changes are picked up
import py_compile
py_compile.compile('run_proxima_demo.py', doraise=True)
py_compile.compile('execution/execution_mapper.py', doraise=True)
print("compile OK", flush=True)
