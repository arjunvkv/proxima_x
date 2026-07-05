import { useMemo } from 'react'
import Card from '../components/ui/Card'
import StatBar from '../components/ui/StatBar'
import VectorField from '../components/charts/VectorField'
import {
  useRealtimeCore,
  useSystemHealthScore,
  useEngineVector,
  useSnapshotCore,
  useRegimeState,
} from '../core/store/selectors'
import { ENGINE_VECTOR_GROUPS } from '../types/engine'

const statFields = [
  { key: 'alignment', label: 'Alignment', color: 'secondary' as const },
  { key: 'stability', label: 'Stability', color: 'primary' as const },
  { key: 'entropy', label: 'Entropy', color: 'warning' as const },
  { key: 'tpiConfidence', label: 'TPI Confidence', color: 'secondary' as const },
  { key: 'shadowAlignment', label: 'Shadow Align', color: 'shadow' as const },
  { key: 'sofScore', label: 'SOF Score', color: 'tpi' as const },
  { key: 'systemIntegrity', label: 'System Integrity', color: 'primary' as const },
  { key: 'killSwitchPressure', label: 'Kill Switch', color: 'alert' as const },
  { key: 'riskExposure', label: 'Risk Exposure', color: 'regime' as const },
  { key: 'rolloutProgress', label: 'Rollout Progress', color: 'warning' as const },
  { key: 'executionIntensity', label: 'Exec Intensity', color: 'regime' as const },
  { key: 'regimeState', label: 'Regime State', color: 'secondary' as const },
]

const logLines = [
  'INITIALIZING PROXIMA CORE... DONE',
  'FETCHING REGIME_STATE: ACTIVE (Confidence 0.88)',
  'TPI_FLOW: DETECTING ASYMMETRY IN MARKETS',
  'SOF_CALCULATION: 0.741 (NOMINAL)',
  'SHADOW_ALIGNMENT: SYNCING WITH CLOUD_RESERVE',
  'TELEMETRY_GRID_REFRESH: OK',
  'PHASE_SHIFT: COLLECTING_EVIDENCE',
  'MEMORY_PURGE: SUCCESSFUL',
  'HEARTBEAT_PULSE: 42ms',
  'ENTROPY_MEASUREMENT: DECREASING',
  'SYSTEM_INTEGRITY: 100%',
  'ENGINE_VECTOR_UPDATE: SUCCESS',
]

function regimeLabel(val: string | number): string {
  if (typeof val === 'string') return val.toUpperCase()
  if (val < 0.2) return 'IDLE'
  if (val < 0.4) return 'LISTEN'
  if (val < 0.5) return 'ANALYZE'
  if (val < 0.7) return 'ACTIVE'
  if (val < 0.9) return 'SURGE'
  return 'CHAOTIC'
}

export default function MainDashboard() {
  const {
    engineVector,
    systemIntegrity,
    killSwitchPressure,
    executionIntensity,
    regimeState,
  } = useRealtimeCore()

  const { balance, profit, equity } = useSnapshotCore()
  const healthScore = useSystemHealthScore()

  // Compute values for stat bars from stores and derived
  const statValues = useMemo(() => ({
    alignment: engineVector[0] ?? 0,
    stability: engineVector[1] ?? 0,
    entropy: engineVector[2] ?? 0,
    tpiConfidence: engineVector[3] ?? 0,
    shadowAlignment: engineVector[4] ?? 0,
    sofScore: engineVector[5] ?? 0,
    systemIntegrity,
    killSwitchPressure,
    riskExposure: engineVector[6] ?? 0,
    rolloutProgress: engineVector[7] ?? 0,
    executionIntensity,
    ...(typeof regimeState === 'number' ? { regimeState } : { regimeState: 0.5 }),
  }), [engineVector, systemIntegrity, killSwitchPressure, executionIntensity, regimeState])

  const profitColor = profit >= 0 ? 'text-primary' : 'text-alert'
  const perfPct = balance > 0 ? ((profit / (balance - profit)) * 100) : 0

  // P&L sparkline data (simulated for now)
  const sparklineData = useMemo(() => {
    const pts = 50
    const base = equity || balance || 10000
    return Array.from({ length: pts }, (_, i) =>
      base + Math.sin(i * 0.3) * 200 + Math.cos(i * 0.7) * 100 + (Math.random() - 0.5) * 100
    )
  }, [equity, balance])

  const minPL = Math.min(...sparklineData)
  const maxPL = Math.max(...sparklineData)
  const range = maxPL - minPL || 1

  // Build sparkline SVG path
  const sparkPath = useMemo(() => {
    const w = 400, h = 150
    return sparklineData.map((v, i) => {
      const x = (i / (sparklineData.length - 1)) * w
      const y = h - ((v - minPL) / range) * h
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
    }).join(' ')
  }, [sparklineData])

  const areaPath = `${sparkPath} L400,150 L0,150 Z`

  return (
    <div className="flex flex-col gap-3 h-full">
      {/* Main content row: 60/40 split */}
      <div className="flex flex-col lg:flex-row gap-3 flex-1 min-h-0">
        {/* LEFT PANEL: Live Telemetry */}
        <div className="lg:w-[60%] flex flex-col gap-3">
          <Card title="Live Telemetry" titleRight={<span className="text-[10px] font-data text-gray-500">REFRESH_RATE: 30fps</span>}>
            {/* 12 Stat Bars in a grid */}
            <div className="grid grid-cols-2 lg:grid-cols-3 gap-3 mb-4">
              {statFields.map((field) => (
                <StatBar
                  key={field.key}
                  label={field.label}
                  value={statValues[field.key as keyof typeof statValues] ?? 0}
                  color={field.color as any}
                  size="sm"
                />
              ))}
            </div>

            {/* Engine Vector Grid */}
            <div className="mt-2">
              <div className="text-[10px] font-data text-gray-500 mb-2 uppercase flex justify-between">
                <span>Engine Vector Grid</span>
                <span>v.1.0.4_alpha</span>
              </div>
              <div className="bg-black/40 border border-primary/5 rounded p-3">
                <VectorField
                  vector={engineVector}
                  width={600}
                  height={100}
                  groups={ENGINE_VECTOR_GROUPS.map(g => ({
                    label: g.label,
                    color: g.color,
                    indices: g.indices,
                  }))}
                />
                <div className="flex justify-between mt-2 text-[8px] font-data uppercase text-gray-500">
                  {ENGINE_VECTOR_GROUPS.map((g) => (
                    <span key={g.label} style={{ color: g.color }}>{g.label}</span>
                  ))}
                </div>
              </div>
            </div>
          </Card>
        </div>

        {/* RIGHT PANEL: P&L Monitoring */}
        <div className="lg:w-[40%] flex flex-col gap-3">
          <Card title="P&L Monitoring" glow="green" className="flex-1">
            <div className="flex flex-col h-full">
              {/* P&L Header */}
              <div className="flex justify-between items-start mb-4">
                <div />
                <div className="flex flex-col items-end">
                  <div className={`text-2xl font-headline font-bold glow-text ${profitColor} leading-none`}>
                    {profit >= 0 ? '+' : ''}${Math.abs(profit).toFixed(2)}
                  </div>
                  <div className="text-[10px] font-data text-primary mt-1">
                    {perfPct >= 0 ? '+' : ''}{perfPct.toFixed(1)}% PERFORMANCE
                  </div>
                </div>
              </div>

              {/* Sparkline */}
              <div className="flex-1 flex flex-col justify-center py-4">
                <svg className="w-full h-36" viewBox="0 0 400 150" preserveAspectRatio="none">
                  <defs>
                    <linearGradient id="pnlGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#00ff88" stopOpacity="0.3" />
                      <stop offset="100%" stopColor="#00ff88" stopOpacity="0" />
                    </linearGradient>
                  </defs>
                  <path
                    d={areaPath}
                    fill="url(#pnlGradient)"
                  />
                  <path
                    d={sparkPath}
                    fill="none"
                    stroke="#00ff88"
                    strokeWidth="2"
                    className="animate-pulse-slow"
                  />
                </svg>
              </div>

              {/* Footer info */}
              <div className="mt-auto pt-4 border-t border-primary/10 flex justify-between items-end">
                <div>
                  <div className="text-[10px] font-data text-gray-500 uppercase">Regime State</div>
                  <div className="text-primary font-headline font-bold tracking-widest">
                    {regimeLabel(regimeState)}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-[10px] font-data text-gray-500 uppercase">Health Score</div>
                  <div className="text-primary font-data text-lg font-bold">
                    {healthScore.toFixed(3)}
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </div>
      </div>

      {/* BOTTOM: System Log */}
      <div className="h-40 bg-black/80 border border-border rounded overflow-y-auto custom-scrollbar p-3">
        <div className="flex items-center gap-2 mb-2">
          <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
          <span className="text-[10px] font-data text-primary/60">SYSTEM_LOG_DUMP_v1.0.4</span>
        </div>
        <pre className="font-data text-[10px] leading-relaxed text-primary/70">
          {logLines.map((line, i) => (
            <div key={i}>{line}</div>
          ))}
        </pre>
      </div>
    </div>
  )
}
