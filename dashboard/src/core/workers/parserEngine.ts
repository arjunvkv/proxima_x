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
  ParsedDashboard,
} from '../../types/domain'

export type SectionType =
  | 'ASSETS'
  | 'FUNNEL'
  | 'ENTROPY'
  | 'THERMO'
  | 'SHADOW'
  | 'EXECUTION'
  | 'META'
  | null

export function detectSection(line: string): SectionType {
  const l = line.toLowerCase()
  if (l.includes('asset evaluation') || l.includes('symbol')) return 'ASSETS'
  if (l.includes('funnel') || l.includes('signal funnel')) return 'FUNNEL'
  if (l.includes('entropy') || l.includes('compression')) return 'ENTROPY'
  if (l.includes('thermo') || l.includes('tick thermo')) return 'THERMO'
  if (l.includes('shadow') || l.includes('mirror')) return 'SHADOW'
  if (l.includes('execution') || l.includes('topology')) return 'EXECUTION'
  if (l.includes('meta') || l.includes('convergence') || l.includes('director') || l.includes('rcl') || l.includes('session')) return 'META'
  return null
}

function extractKeyValue(line: string): { key: string; value: string } | null {
  const match = line.match(/^[\s]*([\w\s/]+?)[:\s]+(.+)$/)
  if (!match) return null
  return {
    key: match[1].trim().toLowerCase().replace(/\s+/g, '_'),
    value: match[2].trim(),
  }
}

function parseNumber(val: string): number {
  const cleaned = val.replace(/[$,%]/g, '')
  const n = parseFloat(cleaned)
  return isNaN(n) ? 0 : n
}

function parseAssetLine(line: string): Partial<SymbolEval> | null {
  const parts = line.split(/\s{2,}|\t/)
  if (parts.length < 3) return null
  const first = parts[0].trim()
  if (!first || first.length > 10) return null
  return {
    symbol: first,
    price: parseNumber(parts[1] ?? '0'),
    spread: parseNumber(parts[2] ?? '0'),
  }
}

export function parseDashboard(raw: string): ParsedDashboard {
  const lines = raw.split('\n')
  let currentSection: SectionType = null
  let insideTable = false

  const assets: SymbolEval[] = []
  const funnel: Partial<FunnelState> = {}
  const entropy: Partial<EntropySnapshot> = {}
  const shadow: Partial<ShadowSnapshot> = {}
  const execution: Partial<ExecutionTopology> = {}
  const director: Partial<DirectorPipeline> = {}
  const rcl: Partial<RclDashboard> = {}
  const session: Partial<SessionBalance> = {}
  const health: Partial<SystemHealth> = {}

  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed) {
      insideTable = false
      continue
    }

    const section = detectSection(trimmed)
    if (section) {
      currentSection = section
      insideTable = false
      continue
    }

    const kv = extractKeyValue(trimmed)

    switch (currentSection) {
      case 'ASSETS': {
        const asset = parseAssetLine(trimmed)
        if (asset && asset.symbol) {
          assets.push(asset as SymbolEval)
          insideTable = true
        }
        break
      }

      case 'FUNNEL': {
        if (kv) {
          if (kv.key.includes('generated')) funnel.generated = parseNumber(kv.value)
          if (kv.key.includes('threshold')) funnel.threshold_passed = parseNumber(kv.value)
          if (kv.key.includes('submitted')) funnel.submitted = parseNumber(kv.value)
          if (kv.key.includes('accepted')) funnel.accepted = parseNumber(kv.value)
          if (kv.key.includes('opened')) funnel.opened = parseNumber(kv.value)
          if (kv.key.includes('leakage')) funnel.leakage_pct = parseNumber(kv.value)
          if (kv.key.includes('pass_rate') || kv.key.includes('pass rate')) funnel.pass_rate = parseNumber(kv.value)
        }
        break
      }

      case 'ENTROPY': {
        if (kv) {
          if (kv.key.includes('entropy_level') || kv.key.includes('entropy level')) entropy.entropy_level = parseNumber(kv.value)
          if (kv.key.includes('derivative')) entropy.entropy_derivative = parseNumber(kv.value)
          if (kv.key.includes('compression')) entropy.compression_ratio = parseNumber(kv.value)
          if (kv.key.includes('signal')) entropy.signal_entropy = parseNumber(kv.value)
          if (kv.key.includes('noise')) entropy.noise_floor_estimate = parseNumber(kv.value)
          if (kv.key.includes('predictability')) entropy.predictability_index = parseNumber(kv.value)
        }
        break
      }

      case 'SHADOW': {
        if (kv) {
          if (kv.key.includes('alignment')) shadow.shadow_alignment = parseNumber(kv.value)
          if (kv.key.includes('sof')) shadow.sof_score = parseNumber(kv.value)
          if (kv.key.includes('edge_decay') || kv.key.includes('edge decay')) shadow.edge_decay = parseNumber(kv.value)
          if (kv.key.includes('preservation')) shadow.edge_preservation = parseNumber(kv.value)
          if (kv.key.includes('efficiency')) shadow.execution_efficiency = parseNumber(kv.value)
        }
        break
      }

      case 'EXECUTION': {
        if (kv) {
          if (kv.key.includes('density')) execution.signal_density = parseNumber(kv.value)
          if (kv.key.includes('fill')) execution.fill_ratio = parseNumber(kv.value)
          if (kv.key.includes('slippage')) execution.slippage_proxy = parseNumber(kv.value)
          if (kv.key.includes('exposure')) execution.risk_exposure = parseNumber(kv.value)
        }
        break
      }

      case 'META': {
        if (kv) {
          if (kv.key.includes('evidence') || kv.key.includes('strength')) director.evidence_strength = parseNumber(kv.value)
          if (kv.key.includes('research_conf') || kv.key.includes('research confidence')) director.research_confidence = parseNumber(kv.value)
          if (kv.key.includes('deployment_conf')) director.deployment_confidence = parseNumber(kv.value)
          if (kv.key.includes('recommendation')) director.recommendation = kv.value
          if (kv.key.includes('classification')) director.classification = kv.value
          if (kv.key.includes('stability')) health.stability_score = parseNumber(kv.value)
          if (kv.key.includes('integrity')) health.system_integrity = parseNumber(kv.value)
          if (kv.key.includes('rollout')) health.rollout_progress = parseNumber(kv.value)
          if (kv.key.includes('kill_switch') || kv.key.includes('kill switch') || kv.key.includes('kill switch pressure')) health.kill_switch_pressure = parseNumber(kv.value)
          if (kv.key.includes('phase')) health.phase = kv.value
        }
        break
      }
    }
  }

  const parsed: ParsedDashboard = {
    assets,
    signalFunnel: {
      generated: funnel.generated ?? 0,
      threshold_passed: funnel.threshold_passed ?? 0,
      triggered: funnel.triggered ?? 0,
      submitted: funnel.submitted ?? 0,
      accepted: funnel.accepted ?? 0,
      opened: funnel.opened ?? 0,
      closed: funnel.closed ?? 0,
      blocked: funnel.blocked ?? 0,
      rejected: funnel.rejected ?? 0,
      timeout: funnel.timeout ?? 0,
      pass_rate: funnel.pass_rate ?? 0,
      submit_rate: funnel.submit_rate ?? 0,
      open_rate: funnel.open_rate ?? 0,
      leakage_pct: funnel.leakage_pct ?? 0,
    },
    entropy: {
      entropy_level: entropy.entropy_level ?? 0,
      entropy_derivative: entropy.entropy_derivative ?? 0,
      compression_ratio: entropy.compression_ratio ?? 0,
      signal_entropy: entropy.signal_entropy ?? 0,
      noise_floor_estimate: entropy.noise_floor_estimate ?? 0,
      predictability_index: entropy.predictability_index ?? 0,
    },
    thermodynamics: {
      tickEntropy: entropy.entropy_level ?? 0,
      signalEnergy: entropy.signal_entropy ?? 0,
      noiseFloor: entropy.noise_floor_estimate ?? 0,
      predictability: entropy.predictability_index ?? 0,
    },
    shadow: {
      shadow_alignment: shadow.shadow_alignment ?? 0,
      sof_score: shadow.sof_score ?? 0,
      edge_decay: shadow.edge_decay ?? 0,
      mirror_divergence: shadow.mirror_divergence ?? 0,
      alpha_transfer_rate: shadow.alpha_transfer_rate ?? 0,
      false_signal_rate: shadow.false_signal_rate ?? 0,
      edge_preservation: shadow.edge_preservation ?? 0,
      execution_efficiency: shadow.execution_efficiency ?? 0,
      winner: shadow.winner ?? 'NONE',
    },
    execution: {
      signal_density: execution.signal_density ?? 0,
      execution_rate: execution.execution_rate ?? 0,
      fill_ratio: execution.fill_ratio ?? 0,
      slippage_proxy: execution.slippage_proxy ?? 0,
      win_rate_proxy: execution.win_rate_proxy ?? 0,
      risk_exposure: execution.risk_exposure ?? 0,
      rotation_events: execution.rotation_events ?? 0,
      lock_events: execution.lock_events ?? 0,
      migration_events: execution.migration_events ?? 0,
    },
    meta: {
      regime: {
        regime_state: 0,
        regime_transition_pressure: 0,
        regime_entropy_gradient: 0,
        regime_stability_velocity: 0,
        per_symbol_regime: {},
      },
      director: director as DirectorPipeline,
      rcl: rcl as RclDashboard,
      session: session as SessionBalance,
      health: health as SystemHealth,
      deployment: {
        asr: 0,
        execution_quality: 'unknown',
        mean_slippage_pts: 0,
        score_trend: 'stable',
        classification: 'unknown',
      },
      dpl: {
        total_snapshots: 0,
        resolved: 0,
        pct_resolved: 0,
        symbols: [],
        regime_distribution: {},
      },
    },
  }

  return parsed
}
