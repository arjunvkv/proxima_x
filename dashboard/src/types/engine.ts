// ── Engine Vector & Derived Metrics ──

export const ENGINE_VECTOR_DIMENSIONS = 32

export interface EngineVectorGroup {
  label: string
  color: string
  indices: [number, number] // start, end (exclusive)
}

export const ENGINE_VECTOR_GROUPS: EngineVectorGroup[] = [
  { label: 'Regime', color: '#ff8800', indices: [0, 5] },
  { label: 'Thermodynamics', color: '#ff00ff', indices: [5, 10] },
  { label: 'TPI/Flow', color: '#00ffff', indices: [10, 15] },
  { label: 'Execution', color: '#0088ff', indices: [15, 20] },
  { label: 'Shadow/Edge', color: '#aa66ff', indices: [20, 25] },
  { label: 'Health', color: '#00ff88', indices: [25, 32] },
]

export function getVectorGroup(index: number): EngineVectorGroup | undefined {
  return ENGINE_VECTOR_GROUPS.find(g => index >= g.indices[0] && index < g.indices[1])
}

export function getGroupValues(vector: number[], group: EngineVectorGroup): number[] {
  return vector.slice(group.indices[0], group.indices[1])
}

export interface SystemMetrics {
  alignment: number
  stability: number
  entropy: number
  tpiConfidence: number
  shadowAlignment: number
  sofScore: number
  systemIntegrity: number
  killSwitchPressure: number
  riskExposure: number
  rolloutProgress: number
  executionIntensity: number
  regimeState: number
}
