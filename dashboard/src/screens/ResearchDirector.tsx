import Card from '../components/ui/Card'
import StatBar from '../components/ui/StatBar'
import Badge from '../components/ui/Badge'
import Panel from '../components/layout/Panel'
import Grid from '../components/layout/Grid'
import { useDashboardDirector, useDashboardHealth, useDashboardSession, useDashboardRcl, useDashboardDeployment } from '../core/store/selectors'

export default function ResearchDirector() {
  const director = useDashboardDirector()
  const health = useDashboardHealth()
  const session = useDashboardSession()
  const rcl = useDashboardRcl()
  const deployment = useDashboardDeployment()

  return (
    <div className="space-y-3">
      <h2 className="font-headline text-sm text-primary uppercase tracking-widest">
        Research Director & Analytics Dashboard
      </h2>

      <Grid cols={3}>
        <Panel title="RCL Dashboard">
          <div className="space-y-2">
            <StatDisplay label="H5 Resolved" value={rcl?.h5_resolved ?? 0} />
            <StatDisplay label="H20 Resolved" value={rcl?.h20_resolved ?? 0} />
            <StatDisplay label="H5 Win Rate" value={rcl ? `${(rcl.h5_win_rate * 100).toFixed(1)}%` : '0.0%'} />
            <StatDisplay label="H20 Win Rate" value={rcl ? `${(rcl.h20_win_rate * 100).toFixed(1)}%` : '0.0%'} />
            <StatDisplay label="Divergence" value={rcl?.divergence ?? 0} />
          </div>
        </Panel>

        <Panel title="Alpha Transfer Analysis">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-gray-400">Research → Paper</span>
              <Badge variant={director ? 'success' : 'warning'}>
                {director ? 'STREAMING' : 'COLLECTING_DATA'}
              </Badge>
            </div>
            <StatDisplay label="Alpha Transfer" value={director?.alpha_transfer ?? 0} />
            <StatDisplay label="Research Conf" value={director?.research_confidence ?? 0} />
            <StatDisplay label="Deployment Conf" value={director?.deployment_confidence ?? 0} />
          </div>
        </Panel>

        <Panel title="System Phase">
          <div className="space-y-2">
            <Badge variant={health ? 'success' : 'warning'}>{health?.phase ?? 'COLLECTING_EVIDENCE'}</Badge>
            <div className="pt-2">
              <StatDisplay label="Stability" value={health?.stability_score ?? 0} />
              <StatDisplay label="Integrity" value={health?.system_integrity ?? 0} />
              <StatDisplay label="Rollout" value={health?.rollout_progress ?? 0} />
            </div>
          </div>
        </Panel>
      </Grid>

      <Card title="Research vs Paper Comparison">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-[10px] font-data">
            <thead>
              <tr className="text-gray-500 border-b border-border">
                <th className="p-2 font-normal">Metric</th>
                <th className="p-2 font-normal">Research</th>
                <th className="p-2 font-normal">Paper</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/50">
              <tr>
                <td className="p-2 text-gray-400">Evidence Strength</td>
                <td className="p-2 text-primary">{director?.evidence_strength.toFixed(3) ?? '—'}</td>
                <td className="p-2">—</td>
              </tr>
              <tr>
                <td className="p-2 text-gray-400">Alpha Transfer</td>
                <td className="p-2 text-primary">{director?.alpha_transfer.toFixed(3) ?? '—'}</td>
                <td className="p-2">{deployment?.asr.toFixed(3) ?? '—'}</td>
              </tr>
              <tr>
                <td className="p-2 text-gray-400">Research Confidence</td>
                <td className="p-2 text-primary">{director?.research_confidence.toFixed(3) ?? '—'}</td>
                <td className="p-2">{deployment?.mean_slippage_pts.toFixed(3) ?? '—'}</td>
              </tr>
              <tr>
                <td className="p-2 text-gray-400">Deployment Confidence</td>
                <td className="p-2 text-primary">{director?.deployment_confidence.toFixed(3) ?? '—'}</td>
                <td className="p-2">{deployment?.score_trend ?? '—'}</td>
              </tr>
              <tr>
                <td className="p-2 text-gray-400">Classification</td>
                <td className="p-2">{director?.classification ?? '—'}</td>
                <td className="p-2">{deployment?.classification ?? '—'}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function StatDisplay({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex justify-between items-center text-[10px]">
      <span className="text-gray-400">{label}</span>
      <span className="font-data text-primary">
        {typeof value === 'number' ? value.toFixed(3) : value}
      </span>
    </div>
  )
}
