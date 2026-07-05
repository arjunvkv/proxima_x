// ── Domain Models for the Proxima Ops Dashboard ──

export interface SymbolEval {
  symbol: string
  price: number
  spread: number
  ecdf_rank: number
  es_val: number
  es_rank: number
  at_rank: number
  sizing_mult: number
  regime: string
  status: string
  entropy: number
  prod_signal?: number
  p_cont?: number
  oss_ev?: number
  oss_conf?: number
  expected_move?: number
  research_drift?: number
  exec_drift?: number
}

export interface FunnelMetrics {
  stage: string
  count: number
  rate: number
}

export interface TpiSnapshot {
  per_symbol: Record<string, {
    tpi: number
    direction: string
    confidence: number
    session: string
    eligible: boolean
    alignment: string
    persistence: number
    curvature: number
  }>
  base_wins: number
  base_losses: number
  aligned_wins: number
  aligned_losses: number
  conflict_wins: number
  conflict_losses: number
  veto_avoided_losses: number
  gates_passed: number
}

export interface RegimeSnapshot {
  regime_state: number
  regime_transition_pressure: number
  regime_entropy_gradient: number
  regime_stability_velocity: number
  per_symbol_regime: Record<string, string>
}

export interface EntropySnapshot {
  entropy_level: number
  entropy_derivative: number
  compression_ratio: number
  signal_entropy: number
  noise_floor_estimate: number
  predictability_index: number
}

export interface ExecutionTopology {
  signal_density: number
  execution_rate: number
  fill_ratio: number
  slippage_proxy: number
  win_rate_proxy: number
  risk_exposure: number
  rotation_events: number
  lock_events: number
  migration_events: number
}

export interface ShadowSnapshot {
  shadow_alignment: number
  sof_score: number
  edge_decay: number
  mirror_divergence: number
  alpha_transfer_rate: number
  false_signal_rate: number
  edge_preservation: number
  execution_efficiency: number
  winner: string
}

export interface DirectorPipeline {
  evidence_strength: number
  research_confidence: number
  deployment_confidence: number
  alpha_transfer: number
  biggest_risk: string
  biggest_strength: string
  recommendation: string
  classification: string
}

export interface RclDashboard {
  h5_resolved: number
  h20_resolved: number
  h5_wins: number
  h20_wins: number
  h5_win_rate: number
  h20_win_rate: number
  divergence: number
}

export interface SessionBalance {
  asia: number
  london: number
  overlap: number
  ny: number
  dead: number
  total: number
  imbalance: number
  status: string
}

export interface SystemHealth {
  stability_score: number
  kill_switch_pressure: number
  rollout_progress: number
  system_integrity: number
  deployment_score: number
  deployment_classification: string
  runtime_hours: number
  runtime_minutes: number
  phase: string
}

export interface DeploymentReality {
  asr: number
  execution_quality: string
  mean_slippage_pts: number
  score_trend: string
  classification: string
}

export interface DplValidation {
  total_snapshots: number
  resolved: number
  pct_resolved: number
  symbols: string[]
  regime_distribution: Record<string, number>
}

export interface FunnelState {
  generated: number
  threshold_passed: number
  triggered: number
  submitted: number
  accepted: number
  opened: number
  closed: number
  blocked: number
  rejected: number
  timeout: number
  pass_rate: number
  submit_rate: number
  open_rate: number
  leakage_pct: number
}

export interface ParsedDashboard {
  assets: SymbolEval[]
  signalFunnel: FunnelState
  entropy: EntropySnapshot
  thermodynamics: {
    tickEntropy: number
    signalEnergy: number
    noiseFloor: number
    predictability: number
  }
  shadow: ShadowSnapshot
  execution: ExecutionTopology
  meta: {
    regime: RegimeSnapshot
    director: DirectorPipeline
    rcl: RclDashboard
    session: SessionBalance
    health: SystemHealth
    deployment: DeploymentReality
    dpl: DplValidation
  }
}
