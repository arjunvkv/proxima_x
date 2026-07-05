import { create } from 'zustand'
import type { SymbolLiveState, LiveFramePayload } from '../../types/ws'
import { ENGINE_VECTOR_DIMENSIONS } from '../../types/engine'

export interface RealtimeState {
  engineVector: number[]
  symbols: Record<string, SymbolLiveState>
  systemIntegrity: number
  killSwitchPressure: number
  executionIntensity: number
  rolloutProgress: number
  regimeState: string
  regimeConfidence: number
  riskExposure: number
  shadowAlignment: number
  sofScore: number
  tpiGlobal: number
  entropyGlobal: number
  drawdown: number
  tickTime: number
  frameId: number
  cycleCount: number
  alignment: number
  stability: number
  entropy: number
  tpiConfidence: number
}

export interface RealtimeActions {
  updateLiveFrame: (frame: LiveFramePayload) => void
  reset: () => void
}

const initialState: RealtimeState = {
  engineVector: new Array(ENGINE_VECTOR_DIMENSIONS).fill(0),
  symbols: {},
  systemIntegrity: 1,
  killSwitchPressure: 0,
  executionIntensity: 0,
  rolloutProgress: 0,
  regimeState: 'IDLE',
  regimeConfidence: 0,
  riskExposure: 0,
  shadowAlignment: 0,
  sofScore: 0,
  tpiGlobal: 0,
  entropyGlobal: 0,
  drawdown: 0,
  tickTime: 0,
  frameId: 0,
  cycleCount: 0,
  alignment: 0,
  stability: 0,
  entropy: 0,
  tpiConfidence: 0,
}

export const useRealtimeStore = create<RealtimeState & RealtimeActions>((set) => ({
  ...initialState,

  updateLiveFrame: (frame) =>
    set({
      engineVector: frame.engine_vector ?? initialState.engineVector,
      symbols: frame.symbols ?? {},
      systemIntegrity: frame.system_integrity ?? 1,
      killSwitchPressure: frame.kill_switch_pressure ?? 0,
      executionIntensity: frame.execution_intensity ?? 0,
      rolloutProgress: frame.rollout_progress ?? 0,
      regimeState: frame.regime_state ?? 'IDLE',
      regimeConfidence: frame.regime_confidence ?? 0,
      riskExposure: frame.risk_exposure ?? 0,
      shadowAlignment: frame.shadow_alignment ?? 0,
      sofScore: frame.sof_score ?? 0,
      tpiGlobal: frame.tpi_global ?? 0,
      entropyGlobal: frame.entropy_global ?? 0,
      drawdown: frame.drawdown ?? 0,
      tickTime: frame.timestamp ?? Date.now(),
      frameId: frame.frame_id ?? 0,
      cycleCount: frame.cycle_count ?? 0,
      alignment: frame.alignment ?? 0,
      stability: frame.stability ?? 0,
      entropy: frame.entropy ?? 0,
      tpiConfidence: frame.tpi_confidence ?? 0,
    }),

  reset: () => set({ ...initialState }),
}))
