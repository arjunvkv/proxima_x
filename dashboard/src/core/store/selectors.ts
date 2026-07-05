import { useRealtimeStore } from './useRealtimeStore'
import { useSnapshotStore } from './useSnapshotStore'
import { useDashboardStore } from './useDashboardStore'
import { useUIStateStore } from './useUIStateStore'
import type { SymbolLiveState } from '../../types/ws'

// ── Realtime Selectors ──

export const useEngineVector = () => useRealtimeStore((s) => s.engineVector)
export const useSymbols = () => useRealtimeStore((s) => s.symbols)
export const useSymbol = (symbol: string) =>
  useRealtimeStore((s) => s.symbols[symbol] ?? null)
export const useSystemIntegrity = () => useRealtimeStore((s) => s.systemIntegrity)
export const useKillSwitchPressure = () => useRealtimeStore((s) => s.killSwitchPressure)
export const useRegimeState = () => useRealtimeStore((s) => s.regimeState)
export const useShadowAlignment = () => useRealtimeStore((s) => s.shadowAlignment)
export const useSofScore = () => useRealtimeStore((s) => s.sofScore)
export const useTpiGlobal = () => useRealtimeStore((s) => s.tpiGlobal)
export const useRealtimeCore = () =>
  useRealtimeStore((s) => ({
    engineVector: s.engineVector,
    systemIntegrity: s.systemIntegrity,
    killSwitchPressure: s.killSwitchPressure,
    executionIntensity: s.executionIntensity,
    regimeState: s.regimeState,
  }))

// ── Snapshot Selectors ──

export const useAccount = () => useSnapshotStore((s) => s.account)
export const usePositions = () => useSnapshotStore((s) => s.positions)
export const useFunnel = () => useSnapshotStore((s) => s.funnel)
export const useRiskExposure = () => useSnapshotStore((s) => s.riskExposure)
export const useSnapshotCore = () =>
  useSnapshotStore((s) => ({
    balance: s.account.balance,
    equity: s.account.equity,
    profit: s.account.profit,
    riskExposure: s.riskExposure,
    positions: s.positions.length,
  }))

// ── Dashboard Selectors ──

export const useAssets = () => useDashboardStore((s) => s.assets)
export const useDashboardEntropy = () => useDashboardStore((s) => s.entropy)
export const useDashboardShadow = () => useDashboardStore((s) => s.shadow)
export const useDashboardDirector = () => useDashboardStore((s) => s.director)
export const useDashboardHealth = () => useDashboardStore((s) => s.health)
export const useDashboardSession = () => useDashboardStore((s) => s.session)
export const useDashboardExecution = () => useDashboardStore((s) => s.execution)
export const useDashboardDeployment = () => useDashboardStore((s) => s.deployment)
export const useDashboardDpl = () => useDashboardStore((s) => s.dpl)
export const useDashboardRcl = () => useDashboardStore((s) => s.rcl)
export const useDashboardRegime = () => useDashboardStore((s) => s.regime)
export const useDashboardSignalFunnel = () => useDashboardStore((s) => s.signalFunnel)
export const useSnapshotDrawdown = () => useSnapshotStore((s) => s.drawdown)
export const useSnapshotSystem = () => useSnapshotStore((s) => ({
  uptime: s.systemUptime,
  latency: s.systemLatency,
}))

// ── UI State Selectors ──

export const useWsStatus = () => useUIStateStore((s) => s.wsStatus)
export const useIsStale = () => {
  const lastLive = useUIStateStore((s) => s.lastLiveTs)
  return Date.now() - lastLive > 3000
}

// ── Derive System Health ──

export const useSystemHealthScore = () =>
  useRealtimeStore((s) => {
    const vals = [s.alignment, s.stability, s.systemIntegrity, s.tpiConfidence]
    return vals.reduce((a, b) => a + b, 0) / Math.max(vals.length, 1)
  })
