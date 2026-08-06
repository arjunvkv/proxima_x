"""ml/ — lightweight research-state clustering (restored).

This package was silently dropped by a broad ``ml/`` .gitignore entry (same
class of bug as ``mvs/utils``). ``research/pipeline.py`` and
``research/information_discovery/state_constructor.py`` import
``ml.clustering.StateClusterer`` at module load, so the demo could not start
without it. Recreated against scikit-learn's bundled HDBSCAN so the demo's
full import chain (and the real-tape paper run) is unblocked with no heavy
third-party dependency.
"""