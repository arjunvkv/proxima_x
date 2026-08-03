import re
from datetime import datetime, timedelta
from pathlib import Path


def symbol_to_file_safe(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "_", name)


def ensure_month_dir(base_path: Path, symbol: str, year: int, month: int) -> Path:
    dir_path = base_path / symbol_to_file_safe(symbol)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_date_range_for_month(year: int, month: int) -> tuple[datetime, datetime]:
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1)
    else:
        end = datetime(year, month + 1, 1)
    return start, end