import { useRealtimeStore } from '../../core/store/useRealtimeStore'
import { useDashboardStore } from '../../core/store/useDashboardStore'

export function useSymbolEntropyMap(): Record<string, number> {
  const symbols = useRealtimeStore((s) => s.symbols)
  const entropyMap: Record<string, number> = {}
  for (const [sym, data] of Object.entries(symbols)) {
    entropyMap[sym] = data.entropy
  }
  return entropyMap
}

export function useEntropyMetrics() {
  const entropy = useDashboardStore((s) => s.entropy)
  return {
    level: entropy?.entropy_level ?? 0,
    derivative: entropy?.entropy_derivative ?? 0,
    compressionRatio: entropy?.compression_ratio ?? 0,
    signalEntropy: entropy?.signal_entropy ?? 0,
    noiseFloor: entropy?.noise_floor_estimate ?? 0,
    predictability: entropy?.predictability_index ?? 0,
  }
}
