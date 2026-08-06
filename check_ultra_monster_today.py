import sys
sys.path.append('.')
from datetime import datetime, timezone
from backtesting.multi_pair_backtest_engine import MultiPairBacktestEngine
from backtesting.data_provider import MT5Provider
from strategies.ultra_monster import UltraMonsterStrategy

# Today's date range - CORRECTED to 2024
start_date = '2024-08-04'
end_date = '2024-08-05'

print('=== ULTRA MONSTER BACKTEST TODAY ===')
print(f'Period: {start_date} to current time')

try:
    provider = MT5Provider()
    engine = MultiPairBacktestEngine(provider)
    
    strategy = UltraMonsterStrategy()
    results = engine.run_backtest(
        strategy,
        start_date=start_date,
        end_date=end_date,
        initial_balance=100000
    )
    
    trades = results['trades']
    print(f'\nTotal trades: {len(trades)}')
    
    if trades:
        print('\n=== TRADES ===')
        for i, trade in enumerate(trades[-5:], 1):
            entry_time = trade.get('entry_time', 'N/A')
            exit_time = trade.get('exit_time', 'N/A')
            symbol = trade.get('symbol', 'N/A')
            side = trade.get('side', 'N/A')
            pnl = trade.get('net_pnl', 0)
            pips = trade.get('pips', 0)
            print(f'{i}. {symbol} {side} | Entry: {entry_time} | Exit: {exit_time} | PnL: ${pnl:.2f} | Pips: {pips:.1f}')
    
    net_pnl = results["net_pnl"]
    win_rate = results["win_rate"]
    print(f'\nNet PnL: ${net_pnl:.2f}')
    print(f'Win Rate: {win_rate:.1f}%')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()