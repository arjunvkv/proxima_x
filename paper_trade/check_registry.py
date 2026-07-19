"""Check account registry status — which strategies claim which MT5 accounts."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from paper_trade.core import registry


def main():
    claims = registry.list_claims()

    if not claims:
        print("No active account claims.")
        return

    print(f"{'Account':>12s}  {'Strategy':<20s}  {'Age':>8s}  {'PID':>6s}")
    print("-" * 52)
    for login, info in sorted(claims.items()):
        age = f"{info['age_sec']}s"
        print(f"{login:>12s}  {info['strategy']:<20s}  {age:>8s}  {info['pid']:>6d}")

    stale = registry.cleanup()
    if stale:
        print(f"\nCleaned {stale} stale entr{'y' if stale == 1 else 'ies'}.")


if __name__ == "__main__":
    main()
