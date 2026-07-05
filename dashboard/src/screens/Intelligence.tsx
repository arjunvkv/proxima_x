import Card from '../components/ui/Card'
import Badge from '../components/ui/Badge'
import Panel from '../components/layout/Panel'
import Grid from '../components/layout/Grid'
import { useDashboardDirector, useDashboardSession, useDashboardHealth, useDashboardDeployment, useDashboardDpl, useFunnel } from '../core/store/selectors'

export default function Intelligence() {
  const director = useDashboardDirector()
  const session = useDashboardSession()
  const health = useDashboardHealth()
  const deployment = useDashboardDeployment()
  const dpl = useDashboardDpl()
  const funnel = useFunnel()

  return (
    <div className="space-y-3">
      <h2 className="font-headline text-sm text-primary uppercase tracking-widest">
        Advanced Intelligence Panels
      </h2>

      <Grid cols={3}>
        <Panel title="Research Director">
          <div className="space-y-3">
            <StatDisplay label="Evidence" value={director?.evidence_strength ?? 0} />
            <StatDisplay label="Research Conf" value={director?.research_confidence ?? 0} />
            <StatDisplay label="Deployment Conf" value={director?.deployment_confidence ?? 0} />
            <StatDisplay label="Alpha Transfer" value={director?.alpha_transfer ?? 0} />
            <div className="pt-2">
              <Badge variant="warning">{director?.classification ?? 'COLLECTING_DATA'}</Badge>
            </div>
          </div>
        </Panel>

        <Panel title="Deployment Reality">
          <div className="space-y-3">
            <StatDisplay label="ASR" value={deployment?.asr ?? 0} />
            <StatDisplay label="Execution Quality" value={deployment?.execution_quality ?? '—'} />
            <StatDisplay label="Mean Slippage" value={`${(deployment?.mean_slippage_pts ?? 0).toFixed(1)} pts`} />
            <div className="flex justify-between items-center">
              <span className="text-[10px] text-gray-400">Classification</span>
              <Badge variant={deployment?.classification === 'NOMINAL' ? 'success' : 'warning'}>
                {deployment?.classification ?? 'COLLECTING_EVIDENCE'}
              </Badge>
            </div>
          </div>
        </Panel>

        <Panel title="Convergence Metrics">
          <div className="space-y-3">
            <StatDisplay label="ATE" value={director?.alpha_transfer ?? 0} />
            <StatDisplay label="Deploy Score" value={health?.deployment_score ?? 0} />
            <StatDisplay label="Stability" value={health?.stability_score ?? 0} />
            <StatDisplay label="Score Trend" value={deployment?.score_trend ?? '—'} />
          </div>
        </Panel>
      </Grid>

      <Grid cols={3}>
        <Panel title="DPL Validation">
          <div className="space-y-2">
            <StatDisplay label="Snapshots" value={dpl?.total_snapshots ?? 0} />
            <StatDisplay label="Resolved" value={dpl ? `${dpl.resolved} (${(dpl.pct_resolved * 100).toFixed(1)}%)` : '0 (0%)'} />
          </div>
        </Panel>

        <Panel title={`Session Balance (${session?.status ?? 'BUILDING'})`}>
          <div className="grid grid-cols-2 gap-2 text-[10px] font-data">
            <span className="text-gray-400">Asia:</span><span>{session?.asia ?? 0}</span>
            <span className="text-gray-400">London:</span><span>{session?.london ?? 0}</span>
            <span className="text-gray-400">Overlap:</span><span className="text-primary">{session?.overlap ?? 0}</span>
            <span className="text-gray-400">NY:</span><span>{session?.ny ?? 0}</span>
            <span className="text-gray-400">Dead:</span><span>{session?.dead ?? 0}</span>
          </div>
        </Panel>

        <Panel title="Occupancy Leakage Audit">
          <div className="grid grid-cols-3 gap-2 text-center text-[10px]">
            <div>
              <div className="text-gray-400">Blocked</div>
              <div className="text-lg font-data">{funnel?.blocked ?? 0}</div>
            </div>
            <div>
              <div className="text-gray-400">Accepted</div>
              <div className="text-lg font-data">{funnel?.accepted ?? 0}</div>
            </div>
            <div>
              <div className="text-gray-400">Rejected</div>
              <div className="text-lg font-data">{funnel?.rejected ?? 0}</div>
            </div>
          </div>
        </Panel>
      </Grid>
    </div>
  )
}

function StatDisplay({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex justify-between items-center text-[10px]">
      <span className="text-gray-400 uppercase tracking-wider">{label}</span>
      <span className="font-data text-primary">{typeof value === 'number' ? value.toFixed(3) : value}</span>
    </div>
  )
}
