import { useMemo } from 'react'
import Card from '../components/ui/Card'
import StatBar from '../components/ui/StatBar'
import Badge from '../components/ui/Badge'
import { useRealtimeCore, useTpiGlobal, useDashboardShadow, useDashboardDirector, useDashboardHealth } from '../core/store/selectors'

function flowLabel(regime: string): string {
  const r = regime.toLowerCase()
  if (r === 'surge' || r === 'chaotic') return 'MAXIMUM'
  if (r === 'active') return 'ACTIVE'
  if (r === 'analyze') return 'ANALYZE'
  if (r === 'listen') return 'LISTEN'
  return 'IDLE'
}

function flowVariant(regime: string): 'success' | 'warning' | 'info' | 'danger' {
  const r = regime.toLowerCase()
  if (r === 'surge' || r === 'chaotic') return 'danger'
  if (r === 'active') return 'success'
  if (r === 'analyze') return 'info'
  return 'warning'
}

export default function TPI() {
  const { regimeState } = useRealtimeCore()
  const tpiGlobal = useTpiGlobal()
  const shadow = useDashboardShadow()
  const director = useDashboardDirector()
  const health = useDashboardHealth()

  const coherence = useMemo(() => {
    const vals: number[] = []
    if (tpiGlobal !== undefined) vals.push(tpiGlobal)
    if (health?.stability_score !== undefined) vals.push(health.stability_score)
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0
  }, [tpiGlobal, health?.stability_score])

  const asymmetry = shadow?.mirror_divergence ?? 0
  const persistence = director?.evidence_strength ?? 0

  return (
    <div className="space-y-3">
      <h2 className="font-headline text-sm text-primary uppercase tracking-widest">
        TPI Flow Overlay & Meta-State Fusion
      </h2>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <Card title="TPI Global Status" glow="blue">
          <div className="space-y-4 py-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-gray-400 uppercase">Global TPI</span>
              <span className="text-xl font-data text-tpi">{(tpiGlobal * 100).toFixed(1)}%</span>
            </div>
            <StatBar label="TPI Confidence" value={tpiGlobal} color="tpi" size="lg" />
            <div className="flex justify-between items-center pt-2">
              <span className="text-[10px] text-gray-400 uppercase">Regime</span>
              <Badge variant="info">{String(regimeState).toUpperCase()}</Badge>
            </div>
          </div>
        </Card>

        <Card title="Meta-State Fusion">
          <div className="space-y-3 py-2">
            <div className="flex justify-between items-center">
              <span className="text-[10px] text-gray-400 uppercase">Flow State</span>
              <Badge variant={flowVariant(regimeState)}>{flowLabel(regimeState)}</Badge>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[10px] text-gray-400 uppercase">Coherence</span>
              <span className="font-data text-primary">{coherence.toFixed(3)}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[10px] text-gray-400 uppercase">Asymmetry</span>
              <span className="font-data text-warning">{asymmetry.toFixed(3)}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[10px] text-gray-400 uppercase">Persistence</span>
              <span className="font-data text-secondary">{persistence.toFixed(3)}</span>
            </div>
          </div>
        </Card>
      </div>

      <Card title="Per-Symbol Alignment">
        <div className="text-center py-8 text-gray-500 text-xs font-data">
          Symbol alignment table will render here when data is streaming
        </div>
      </Card>
    </div>
  )
}
