import { create } from 'zustand'
import type {
  SymbolEval,
  FunnelState,
  EntropySnapshot,
  ShadowSnapshot,
  ExecutionTopology,
  DirectorPipeline,
  RclDashboard,
  SessionBalance,
  SystemHealth,
  RegimeSnapshot,
  DeploymentReality,
  DplValidation,
} from '../../types/domain'

export interface DashboardState {
  assets: SymbolEval[]
  signalFunnel: FunnelState | null
  entropy: EntropySnapshot | null
  shadow: ShadowSnapshot | null
  execution: ExecutionTopology | null
  regime: RegimeSnapshot | null
  director: DirectorPipeline | null
  rcl: RclDashboard | null
  session: SessionBalance | null
  health: SystemHealth | null
  deployment: DeploymentReality | null
  dpl: DplValidation | null
  lastUpdate: number
}

export interface DashboardActions {
  setParsed: (data: Partial<DashboardState>) => void
  reset: () => void
}

const initialState: DashboardState = {
  assets: [],
  signalFunnel: null,
  entropy: null,
  shadow: null,
  execution: null,
  regime: null,
  director: null,
  rcl: null,
  session: null,
  health: null,
  deployment: null,
  dpl: null,
  lastUpdate: 0,
}

export const useDashboardStore = create<DashboardState & DashboardActions>((set) => ({
  ...initialState,

  setParsed: (data) =>
    set({
      ...data,
      lastUpdate: Date.now(),
    }),

  reset: () => set({ ...initialState }),
}))
