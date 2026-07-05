import { useDataStaleness } from '../../utils/staleDetection'

export default function StaleIndicator() {
  const { stale, secondsSinceLastUpdate } = useDataStaleness()

  if (!stale) return null

  return (
    <div className="fixed bottom-10 right-4 z-50 flex items-center gap-2 bg-warning/20 border border-warning/30 rounded px-3 py-1.5">
      <span className="w-2 h-2 rounded-full bg-warning animate-pulse" />
      <span className="text-[10px] font-data text-warning uppercase tracking-wider">
        DATA STALE — {secondsSinceLastUpdate}s ago
      </span>
    </div>
  )
}
