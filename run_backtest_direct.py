"""Run MT5 backtest directly - fresh Python process each time."""
import subprocess, sys, time, json, re, uuid
from pathlib import Path

MT5_TERMINAL = r"C:\Program Files\MetaTrader 5\terminal64.exe"
BASE_DIR = Path.home() / ".meta-trader-mcp" / "backtests"

PERIOD_MAP = {"M1":1,"M2":2,"M3":3,"M4":4,"M5":5,"M6":6,"M10":10,"M12":12,"M15":15,"M20":20,"M30":30,"H1":60,"H2":120,"H3":180,"H4":240,"H6":360,"H8":480,"H12":720,"D1":1440,"W1":10080,"MN1":43200}

def generate_ini(expert: str, symbol: str, timeframe: str,
                  from_date: str, to_date: str, report_path: str,
                  deposit: float = 10000, leverage: int = 100,
                  model: int = 1) -> str:
    expert = re.sub(r"\.mq5$", "", expert, flags=re.IGNORECASE)
    period = PERIOD_MAP.get(timeframe.upper(), 60)
    lines = [
        "[Tester]",
        f"Expert={expert}",
        f"Symbol={symbol}",
        f"Period={period}",
        f"Model={model}",
        f"FromDate={from_date}",
        f"ToDate={to_date}",
        f"Deposit={deposit:g}",
        f"Leverage={leverage}",
        "Optimization=0",
        f"Report={report_path}",
        "ReplaceReport=1",
        "ShutdownTerminal=1",
    ]
    return "\n".join(lines) + "\n"

def run_backtest(expert: str, symbol: str, timeframe: str = "H1",
                 from_date: str = "", to_date: str = "",
                 deposit: float = 10000, leverage: int = 100,
                 model: int = 1, poll_interval: int = 10,
                 timeout: int = 600) -> dict:
    """Launch MT5 backtest, wait for completion, return parsed result."""
    run_id = uuid.uuid4().hex[:12]
    run_dir = BASE_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "report.htm"
    ini_path = run_dir / "tester.ini"

    ini_path.write_text(
        generate_ini(expert, symbol, timeframe, from_date, to_date,
                     str(report_path), deposit, leverage, model),
        encoding="utf-8"
    )

    print(f"[{run_id}] Launching: {MT5_TERMINAL} /config:{ini_path}")
    proc = subprocess.Popen(
        [MT5_TERMINAL, f"/config:{ini_path}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"[{run_id}] PID={proc.pid}, waiting up to {timeout}s...")

    # Poll for report file or process exit
    start = time.time()
    while time.time() - start < timeout:
        if report_path.exists() and report_path.stat().st_size > 0:
            elapsed = time.time() - start
            print(f"[{run_id}] Report ready at {elapsed:.0f}s")
            # Give MT5 a moment to finish writing
            time.sleep(2)
            text = report_path.read_text(encoding="utf-8", errors="replace")
            metrics = parse_report(text)
            return {"status": "COMPLETED", "run_id": run_id,
                    "elapsed_s": round(elapsed), "report": str(report_path),
                    **metrics}
        # Check if process exited without creating report
        ret = proc.poll()
        if ret is not None:
            elapsed = time.time() - start
            print(f"[{run_id}] Process exited at {elapsed:.0f}s (code={ret})")
            # Try reading report anyway
            if report_path.exists() and report_path.stat().st_size > 0:
                text = report_path.read_text(encoding="utf-8", errors="replace")
                metrics = parse_report(text)
                return {"status": "COMPLETED", "run_id": run_id,
                        "elapsed_s": round(elapsed), "report": str(report_path),
                        **metrics}
            return {"status": "FAILED", "run_id": run_id,
                    "exit_code": ret, "report": str(report_path) if report_path.exists() else None}
        time.sleep(poll_interval)

    # Timeout - kill process
    print(f"[{run_id}] TIMEOUT after {timeout}s - killing process")
    proc.kill()
    proc.wait(timeout=5)
    if report_path.exists() and report_path.stat().st_size > 0:
        text = report_path.read_text(encoding="utf-8", errors="replace")
        metrics = parse_report(text)
        return {"status": "COMPLETED", "run_id": run_id,
                "elapsed_s": timeout, "report": str(report_path), **metrics}
    return {"status": "TIMEOUT", "run_id": run_id, "elapsed_s": timeout}

REPORT_PATTERNS = {
    "net_profit": ("Total Net Profit", "Net Profit"),
    "profit_factor": ("Profit Factor",),
    "max_drawdown": ("Maximal Drawdown", "Max Drawdown", "Balance Drawdown Maximal"),
    "win_rate": ("Profit Trades", "Win Rate"),
    "total_trades": ("Total Trades", "Trades Total"),
}

def parse_report(text: str) -> dict:
    import html
    plain = re.sub(r"<[^>]+>", " ", text)
    plain = re.sub(r"\s+", " ", plain)
    metrics = {}
    for key, labels in REPORT_PATTERNS.items():
        for label in labels:
            match = re.search(
                re.escape(label) + r"[^0-9\-+]{0,40}(-?\d[\d\s,.'']*\d|\d)",
                plain, flags=re.IGNORECASE,
            )
            if match:
                raw = match.group(1)
                cleaned = re.sub(r"[\s,']", "", raw)
                try:
                    metrics[key] = float(cleaned)
                except ValueError:
                    continue
                break
    # Win rate percentage
    m = re.search(r"Profit Trades.*?\((\d+(?:\.\d+)?)\s*%\s*\)", plain, flags=re.IGNORECASE)
    if m:
        metrics["win_rate"] = float(m.group(1))
    return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", default="V2z_trend")
    parser.add_argument("--symbol", default="EURAUD")
    parser.add_argument("--from-date", default="2026.06.08")
    parser.add_argument("--to-date", default="2026.06.09")
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--deposit", type=float, default=10000)
    parser.add_argument("--leverage", type=int, default=100)
    parser.add_argument("--model", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--poll", type=int, default=5)
    args = parser.parse_args()

    result = run_backtest(
        expert=args.expert, symbol=args.symbol,
        timeframe=args.timeframe,
        from_date=args.from_date, to_date=args.to_date,
        deposit=args.deposit, leverage=args.leverage,
        model=args.model, poll_interval=args.poll,
        timeout=args.timeout,
    )
    print(json.dumps(result, indent=2, default=str))
