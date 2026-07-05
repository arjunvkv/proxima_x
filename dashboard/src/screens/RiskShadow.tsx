import Card from '../components/ui/Card'
import Gauge from '../components/ui/Gauge'
import StatBar from '../components/ui/StatBar'
import Badge from '../components/ui/Badge'
import {
  useKillSwitchPressure,
  useShadowAlignment,
  useSofScore,
  useRiskExposure,
  useSystemIntegrity,
  useDashboardShadow,
  useDashboardExecution,
  useDashboardDeployment,
  useDashboardDpl,
  useSnapshotDrawdown,
  useSnapshotSystem,
  useAccount,
} from '../core/store/selectors'

export default function RiskShadow() {
  const killSwitch = useKillSwitchPressure()
  const shadowAlign = useShadowAlignment()
  const sofScore = useSofScore()
  const riskExposure = useRiskExposure()
  const integrity = useSystemIntegrity()
  const shadow = useDashboardShadow()
  const execution = useDashboardExecution()
  const deployment = useDashboardDeployment()
  const dpl = useDashboardDpl()
  const drawdown = useSnapshotDrawdown()
  const sys = useSnapshotSystem()
  const account = useAccount()

  return (
    <div className="space-y-3">
      <h2 className="font-headline text-sm text-primary uppercase tracking-widest">
        Risk Hardening / Shadow System / Execution Bridge
      </h2>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <Card title="Risk Exposure" glow="green">
          <div className="flex justify-center py-4">
            <Gauge value={1 - riskExposure} label="Safety" color="#00ff88" size="lg" threshold={0.5} />
          </div>
        </Card>

        <Card title="Kill Switch" glow="blue">
          <div className="flex justify-center py-4">
            <Gauge value={killSwitch} label="Pressure" color="#ff4444" size="lg" threshold={0.7} />
          </div>
        </Card>

        <Card title="Shadow Align">
          <div className="flex justify-center py-4">
            <Gauge value={shadowAlign} label="Alignment" color="#aa66ff" size="lg" />
          </div>
        </Card>

        <Card title="SOF Score">
          <div className="flex justify-center py-4">
            <Gauge value={sofScore} label="SOF" color="#00ffff" size="lg" />
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        <Card title="Risk Metrics">
          <div className="space-y-2">
            <StatBar label="Kill Switch Pressure" value={killSwitch} color="alert" size="md" />
            <StatBar label="Risk Exposure" value={riskExposure} color="regime" size="md" />
            <StatBar label="System Integrity" value={integrity} color="primary" size="md" />
            <StatBar label="Drawdown" value={drawdown} color="alert" size="md" />
            <StatBar label="Shadow Alignment" value={shadowAlign} color="shadow" size="md" />
          </div>
        </Card>

        <Card title="Shadow System">
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-[10px] text-gray-400 uppercase">Edge Preservation</span>
              <span className="font-data text-primary">
                {(shadow?.edge_preservation ?? 0).toFixed(3)}
              </span>
            </div>
            <StatBar label="Edge Decay" value={shadow?.edge_decay ?? 0} color="warning" size="md" />
            <StatBar label="Alpha Transfer" value={shadow?.alpha_transfer_rate ?? 0} color="secondary" size="md" />
            <StatBar label="False Signal Rate" value={shadow?.false_signal_rate ?? 0} color="alert" size="md" />
            <StatBar label="Mirror Divergence" value={shadow?.mirror_divergence ?? 0} color="regime" size="md" />
            <div className="flex justify-between items-center pt-1">
              <span className="text-[10px] text-gray-400 uppercase">Winner</span>
              <Badge variant={shadow?.winner === 'shadow' ? 'purple' : 'info'}>
                {shadow?.winner ?? 'NONE'}
              </Badge>
            </div>
          </div>
        </Card>

        <Card title="System Health">
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-[10px] text-gray-400 uppercase">Uptime</span>
              <span className="font-data text-primary">{sys.uptime.toFixed(1)}s</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[10px] text-gray-400 uppercase">Latency</span>
              <span className="font-data text-secondary">{sys.latency}ms</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[10px] text-gray-400 uppercase">Balance</span>
              <span className="font-data text-primary">${account.balance.toFixed(2)}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[10px] text-gray-400 uppercase">Equity</span>
              <span className="font-data text-warning">${account.equity.toFixed(2)}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-[10px] text-gray-400 uppercase">Profit</span>
              <span className="font-data text-tpi">${account.profit.toFixed(2)}</span>
            </div>
          </div>
        </Card>
      </div>

      <Card title="Shadow System Details">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 py-2">
          <div className="text-center">
            <div className="text-[10px] text-gray-400 uppercase">Edge Preservation</div>
            <div className="text-lg font-data text-primary">{(shadow?.edge_preservation ?? 0).toFixed(3)}</div>
          </div>
          <div className="text-center">
            <div className="text-[10px] text-gray-400 uppercase">Alpha Transfer</div>
            <div className="text-lg font-data text-secondary">{(shadow?.alpha_transfer_rate ?? 0).toFixed(3)}</div>
          </div>
          <div className="text-center">
            <div className="text-[10px] text-gray-400 uppercase">False Signal Rate</div>
            <div className="text-lg font-data text-warning">{(shadow?.false_signal_rate ?? 0).toFixed(3)}</div>
          </div>
          <div className="text-center">
            <div className="text-[10px] text-gray-400 uppercase">Winner</div>
            <div className="text-lg font-data text-accent">{shadow?.winner ?? 'NONE'}</div>
          </div>
        </div>
      </Card>

      <Card title="Execution Bridge">
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-4 py-2">
          <div className="text-center">
            <div className="text-[10px] text-gray-400 uppercase">Density</div>
            <div className="text-lg font-data text-primary">{(execution?.signal_density ?? 0).toFixed(3)}</div>
          </div>
          <div className="text-center">
            <div className="text-[10px] text-gray-400 uppercase">Efficiency</div>
            <div className="text-lg font-data text-secondary">{(shadow?.execution_efficiency ?? 0).toFixed(3)}</div>
          </div>
          <div className="text-center">
            <div className="text-[10px] text-gray-400 uppercase">Fill Rate</div>
            <div className="text-lg font-data text-primary">{(execution?.fill_ratio ?? 0).toFixed(3)}</div>
          </div>
          <div className="text-center">
            <div className="text-[10px] text-gray-400 uppercase">Slippage</div>
            <div className="text-lg font-data text-warning">{(deployment?.mean_slippage_pts ?? execution?.slippage_proxy ?? 0).toFixed(3)}</div>
          </div>
          <div className="text-center">
            <div className="text-[10px] text-gray-400 uppercase">Win Rate</div>
            <div className="text-lg font-data text-tpi">{(execution?.win_rate_proxy ?? 0).toFixed(3)}</div>
          </div>
        </div>
        <div className="flex justify-between items-center px-2 pt-3 text-[10px] text-gray-500 uppercase tracking-wider border-t border-border">
          <span>Rotation: {execution?.rotation_events ?? 0}</span>
          <span>Lock: {execution?.lock_events ?? 0}</span>
          <span>Migration: {execution?.migration_events ?? 0}</span>
        </div>
      </Card>

      {dpl && (
        <Card title="MOF Evaluation">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 py-2">
            <div className="text-center">
              <div className="text-[10px] text-gray-400 uppercase">Score</div>
              <div className="text-lg font-data text-primary">{(dpl as any).mof_score ?? 0}</div>
            </div>
            <div className="text-center">
              <div className="text-[10px] text-gray-400 uppercase">Coherence</div>
              <div className="text-lg font-data text-secondary">{(dpl as any).mof_coherence ?? 0}</div>
            </div>
            <div className="text-center">
              <div className="text-[10px] text-gray-400 uppercase">Confidence</div>
              <div className="text-lg font-data text-warning">{(dpl as any).mof_confidence ?? 0}</div>
            </div>
            <div className="text-center">
              <div className="text-[10px] text-gray-400 uppercase">Stability</div>
              <div className="text-lg font-data text-tpi">{(dpl as any).mof_stability ?? 0}</div>
            </div>
          </div>
        </Card>
      )}
    </div>
  )
}
