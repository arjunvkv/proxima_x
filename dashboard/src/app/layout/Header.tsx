import { useSnapshotCore, useRealtimeCore, useWsStatus } from '../../core/store/selectors'

export default function Header() {
  const { balance, profit } = useSnapshotCore()
  const { systemIntegrity } = useRealtimeCore()
  const wsStatus = useWsStatus()

  const statusColor =
    wsStatus === 'STREAMING' ? 'bg-primary' :
    wsStatus === 'CONNECTED' ? 'bg-secondary' :
    wsStatus === 'RECONNECTING' || wsStatus === 'CONNECTING' ? 'bg-warning' :
    'bg-alert'

  const statusText =
    wsStatus === 'STREAMING' ? 'LIVE' :
    wsStatus === 'CONNECTED' ? 'CONNECTED' :
    wsStatus === 'RECONNECTING' ? 'RECONNECTING' :
    wsStatus === 'CONNECTING' ? 'CONNECTING' :
    wsStatus === 'ERROR' ? 'ERROR' :
    'DISCONNECTED'

  const profitColor = profit >= 0 ? 'text-primary' : 'text-alert'

  return (
    <header className="flex items-center justify-between px-4 h-12 bg-background border-b border-border">
      <div className="flex items-center gap-4">
        <span className="font-headline font-bold text-primary tracking-widest text-lg">
          PROXIMA_OPS_v1.0
        </span>
        <div className="hidden md:flex items-center gap-3 border-l border-border pl-4">
          <div className="flex items-center gap-2">
            <span className={`relative flex h-2 w-2`}>
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${statusColor} opacity-75`} />
              <span className={`relative inline-flex rounded-full h-2 w-2 ${statusColor}`} />
            </span>
            <span className="font-data text-xs opacity-70">{statusText}</span>
          </div>
          <div className="bg-primary/10 px-2 py-0.5 rounded text-[10px] font-bold border border-primary/20">
            INTEGRITY: {(systemIntegrity * 100).toFixed(0)}%
          </div>
        </div>
      </div>

      <div className="flex items-center gap-6">
        <div className="hidden lg:flex items-center gap-6 font-data">
          <div className="flex flex-col items-end">
            <span className="text-[10px] opacity-50 uppercase">Balance</span>
            <span className="glow-text text-primary font-bold">
              ${balance.toLocaleString(undefined, { minimumFractionDigits: 2 })}
            </span>
          </div>
          <div className="flex flex-col items-end">
            <span className="text-[10px] opacity-50 uppercase">P&L</span>
            <span className={profitColor}>
              {profit >= 0 ? '+' : ''}${Math.abs(profit).toFixed(2)}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className="material-symbols-outlined text-secondary hover:text-primary transition-colors cursor-pointer">
            account_balance_wallet
          </span>
          <span className="material-symbols-outlined text-secondary hover:text-primary transition-colors cursor-pointer">
            terminal
          </span>
        </div>
      </div>
    </header>
  )
}
