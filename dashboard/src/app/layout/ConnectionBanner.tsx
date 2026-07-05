import { useUIStateStore } from '../../core/store/useUIStateStore'

export default function ConnectionBanner() {
  const wsStatus = useUIStateStore((s) => s.wsStatus)
  const wsAttempts = useUIStateStore((s) => s.wsAttempts)

  if (wsStatus === 'STREAMING' || wsStatus === 'CONNECTED') return null

  const isReconnecting = wsStatus === 'RECONNECTING' || wsStatus === 'CONNECTING'
  const isError = wsStatus === 'ERROR'

  const bgColor = isError ? 'bg-alert' : isReconnecting ? 'bg-warning' : 'bg-alert'
  const text = isReconnecting
    ? `RECONNECTING... Attempt ${wsAttempts}`
    : isError
    ? `CONNECTION ERROR — Retrying...`
    : `DISCONNECTED — Waiting...`

  const pulseClass = isReconnecting ? 'animate-pulse' : ''

  return (
    <div className={`${bgColor} ${pulseClass} px-4 py-1 flex items-center justify-center gap-2`}>
      <span className="w-1.5 h-1.5 rounded-full bg-white" />
      <span className="text-[10px] font-bold text-background uppercase tracking-wider font-data">
        {text}
      </span>
    </div>
  )
}
