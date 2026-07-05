// ── WebSocket Message Contracts ──

export type StreamType = 'live' | 'state' | 'dashboard' | 'replay'

export interface WSBaseMessage<T = unknown> {
  stream: StreamType
  ts: number
  seq?: number
  payload: T
}

// ── Live Frame (30fps) ──

export interface SymbolLiveState {
  price: number
  spread: number
  tpi: number
  entropy: number
  pressure: number
  execution_intensity: number
  regime: string
  alignment: number
}

export interface LiveFramePayload {
  engine_vector: number[]
  system_integrity: number
  kill_switch_pressure: number
  execution_intensity: number
  rollout_progress: number
  regime_state: string
  regime_confidence: number
  risk_exposure: number
  drawdown: number
  symbols: Record<string, SymbolLiveState>
  entropy_global: number
  shadow_alignment: number
  sof_score: number
  tpi_global: number
  tpi_alignment_map: Record<string, number>
  engine_health_vector?: number[]
  cycle_count?: number
  frame_id?: number
  timestamp?: number
  alignment?: number
  stability?: number
  entropy?: number
  tpi_confidence?: number
}

export interface LiveFrameMessage extends WSBaseMessage<LiveFramePayload> {
  stream: 'live'
}

// ── State Snapshot (1fps) ──

export interface AccountState {
  balance: number
  equity: number
  profit: number
  margin: number
}

export interface Position {
  ticket: number
  symbol: string
  side: string
  volume: number
  entry_price: number
  current_price: number
  profit: number
  bars_elapsed: number
  entry_es_rank?: number
  entry_at_rank?: number
  econ_ratio?: number
  expected_move?: number
  trigger_count_while_open: number
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

export interface TelemetrySnapshot {
  account: AccountState
  positions: Position[]
  funnel: FunnelState
  risk: {
    exposure: number
    drawdown: number
    limit: number
  }
  system_health: {
    uptime: number
    latency_ms: number
    error_rate: number
  }
}

export interface SnapshotMessage extends WSBaseMessage<TelemetrySnapshot> {
  stream: 'state'
}

// ── Dashboard Text ──

export interface DashboardMessage extends WSBaseMessage<string> {
  stream: 'dashboard'
  payload: string
}

// ── Replay ──

export interface ReplayRequest {
  type: 'replay_request'
  from: number
  to: number
  stream?: string
  cursor?: number
  max_results?: number
}

export interface ReplayMessage extends WSBaseMessage<{
  frames: LiveFramePayload[]
  snapshots: TelemetrySnapshot[]
  cursor: number
  complete: boolean
}> {
  stream: 'replay'
}

// ── Connection States ──

export type WSConnectionState =
  | 'DISCONNECTED'
  | 'CONNECTING'
  | 'CONNECTED'
  | 'STREAMING'
  | 'RECONNECTING'
  | 'ERROR'

export interface WSConnectionEvent {
  state: WSConnectionState
  timestamp: number
  attempt?: number
  reason?: string
}

// ── Composite Messages ──

export type WSIncomingMessage =
  | LiveFrameMessage
  | SnapshotMessage
  | DashboardMessage
  | ReplayMessage
