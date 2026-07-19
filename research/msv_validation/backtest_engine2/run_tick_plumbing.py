"""DEPRECATED — Tick-level plumbing analysis complete.
See SUMMARY.md for final interpretation.

Key conclusions:
- Spread recovery: vol DROPS after spread shock (structural, use as trade veto)
- Quote asymmetry: 0.7-0.8 catch-up ratio (structural, no directional edge)
- Tick arrival: vol slightly INCREASES after acceleration (not useful for MR)
- Tick states: 99% unclassified (data limitation of Exness Raw_Spread)
- Latency race: too sparse, no lead-lag at 1ms

Decision: Stop raw tick directional signal research.
Keep: spread stress, quote stress as context sensors for Market State Engine.
"""
