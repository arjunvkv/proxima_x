import { useWsStatus, useIsStale, useRealtimeCore } from '../../core/store/selectors'

export default function StatusBar() {
  const wsStatus = useWsStatus()
  const isStale = useIsStale()
  const { systemIntegrity } = useRealtimeCore()

  const pulseClass = wsStatus === 'STREAMING' ? 'bg-primary animate-pulse' :
    isStale ? 'bg-warning animate-pulse' : 'bg-alert'

  return (
    <footer className="h-7 bg-background border-t border-border flex items-center px-4 justify-between text-[10px] uppercase tracking-tighter text-gray-500">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-1">
          <div className={`w-1.5 h-1.5 rounded-full ${pulseClass}`} />
          <span>{wsStatus}</span>
        </div>
        <span>|</span>
        <span>INTEGRITY: {(systemIntegrity * 100).toFixed(0)}%</span>
        <span>|</span>
        <span>PROXIMA OPS v1.0</span>
      </div>
      <div className="font-data opacity-50">
        [WS: {wsStatus}]
      </div>
    </footer>
  )
}
