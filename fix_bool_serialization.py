content = open("proxima_command_center/rolling_backtest_engine.py", "r", encoding="utf-8").read()

# Fix all Python bool occurrences -> JSON-safe int
content = content.replace('"is_win": pnl >= 0,', '"is_win": 1 if pnl >= 0 else 0,')
content = content.replace('"is_live": False,', '"is_live": 0,')
content = content.replace('"is_live": True,', '"is_live": 1,')

open("proxima_command_center/rolling_backtest_engine.py", "w", encoding="utf-8").write(content)

remaining_bool = content.count('"is_win": pnl >= 0')
fixed_int      = content.count('"is_win": 1 if pnl')
print(f"Remaining bool is_win : {remaining_bool}")
print(f"Fixed int  is_win     : {fixed_int}")
print("Done.")
