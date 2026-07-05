import { create } from 'zustand'
import type { TelemetrySnapshot, Position, AccountState, FunnelState } from '../../types/ws'

export interface SnapshotState {
  account: AccountState
  positions: Position[]
  funnel: FunnelState
  riskExposure: number
  drawdown: number
  systemUptime: number
  systemLatency: number
  lastUpdate: number
}

export interface SnapshotActions {
  updateSnapshot: (snapshot: TelemetrySnapshot) => void
  reset: () => void
}

const initialState: SnapshotState = {
  account: {
    balance: 0,
    equity: 0,
    profit: 0,
    margin: 0,
  },
  positions: [],
  funnel: {
    generated: 0,
    threshold_passed: 0,
    triggered: 0,
    submitted: 0,
    accepted: 0,
    opened: 0,
    closed: 0,
    blocked: 0,
    rejected: 0,
    timeout: 0,
    pass_rate: 0,
    submit_rate: 0,
    open_rate: 0,
    leakage_pct: 0,
  },
  riskExposure: 0,
  drawdown: 0,
  systemUptime: 0,
  systemLatency: 0,
  lastUpdate: 0,
}

export const useSnapshotStore = create<SnapshotState & SnapshotActions>((set) => ({
  ...initialState,

  updateSnapshot: (snapshot) =>
    set({
      account: snapshot.account ?? initialState.account,
      positions: snapshot.positions ?? [],
      funnel: snapshot.funnel ?? initialState.funnel,
      riskExposure: snapshot.risk?.exposure ?? 0,
      drawdown: snapshot.risk?.drawdown ?? 0,
      systemUptime: snapshot.system_health?.uptime ?? 0,
      systemLatency: snapshot.system_health?.latency_ms ?? 0,
      lastUpdate: Date.now(),
    }),

  reset: () => set({ ...initialState }),
}))
