import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from replay.environment import build_replay_environment, ReplayConfig
from replay.clock_patcher import patch_clock
from research.metrics import MetricsCollector
from research.layer_config import LayerConfig
from run_proxima_demo import ProximaDemo


def run_ablation(layer_config: LayerConfig, replay_config: ReplayConfig):
    env = build_replay_environment(replay_config)
    patch_clock(env.clock)
    demo = ProximaDemo(env=env, layer_config=layer_config)
    demo._tick_limit = 50000
    try:
        demo.run_demo()
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass
    metrics = MetricsCollector.collect(demo, env)
    metrics["wall_runtime_sec"] = 0.0
    metrics["ticks_processed"] = env.replay_feed.cursor if (env and hasattr(env, 'replay_feed') and env.replay_feed) else 0
    return metrics
