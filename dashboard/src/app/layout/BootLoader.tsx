import { useState, useEffect, type ReactNode } from 'react'
import { useUIStateStore } from '../../core/store/useUIStateStore'

interface Props {
  children: ReactNode
}

export default function BootLoader({ children }: Props) {
  const [booted, setBooted] = useState(false)
  const [hasData, setHasData] = useState(false)
  const wsStatus = useUIStateStore((s) => s.wsStatus)
  const lastLiveTs = useUIStateStore((s) => s.lastLiveTs)
  const lastDashboardTs = useUIStateStore((s) => s.lastDashboardTs)

  if (lastLiveTs > 0 || lastDashboardTs > 0) {
    if (!hasData) setHasData(true)
  }

  useEffect(() => {
    if (hasData && (wsStatus === 'STREAMING' || wsStatus === 'CONNECTED')) {
      const timer = setTimeout(() => setBooted(true), 500)
      return () => clearTimeout(timer)
    }
  }, [wsStatus, hasData])

  if (booted) return <>{children}</>

  return (
    <div className="fixed inset-0 bg-background flex flex-col items-center justify-center z-[9999] terminal-grid">
      <div className="flex flex-col items-center gap-6">
        <div className="text-2xl font-headline font-bold text-primary tracking-widest">
          PROXIMA_OPS_v1.0
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
          <span className="text-xs font-data text-gray-500 uppercase tracking-wider">
            Initializing Engine
          </span>
        </div>
        <div className="w-48 h-1 bg-border rounded overflow-hidden">
          <div className="h-full bg-primary rounded animate-pulse-slow" style={{ width: '60%' }} />
        </div>
        <div className="text-[10px] font-data text-gray-600 uppercase tracking-widest">
          {wsStatus === 'CONNECTING' ? 'Connecting to WebSocket...' :
           wsStatus === 'CONNECTED' ? 'Waiting for data stream...' :
           wsStatus === 'ERROR' ? 'Connection error — retrying...' :
           'Starting...'}
        </div>
      </div>
    </div>
  )
}
