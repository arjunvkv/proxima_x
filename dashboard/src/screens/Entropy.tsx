import Card from '../components/ui/Card'
import Gauge from '../components/ui/Gauge'
import StatBar from '../components/ui/StatBar'
import { useDashboardEntropy } from '../core/store/selectors'

export default function Entropy() {
  const entropy = useDashboardEntropy()

  return (
    <div className="space-y-3">
      <h2 className="font-headline text-sm text-primary uppercase tracking-widest">
        Entropy Compression Heatmap
      </h2>

      <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
        <Card title="Entropy Level" glow="green">
          <div className="flex justify-center py-4">
            <Gauge
              value={entropy?.entropy_level ?? 0}
              label="Entropy"
              color="#00ff88"
              size="lg"
            />
          </div>
        </Card>

        <Card title="Compression Ratio" glow="blue">
          <div className="flex justify-center py-4">
            <Gauge
              value={entropy?.compression_ratio ?? 0}
              label="Compression"
              color="#00ccff"
              size="lg"
            />
          </div>
        </Card>

        <Card title="Predictability">
          <div className="flex justify-center py-4">
            <Gauge
              value={entropy?.predictability_index ?? 0}
              label="Predictability"
              color="#aa66ff"
              size="lg"
            />
          </div>
        </Card>
      </div>

      <Card title="Entropy Metrics">
        <div className="space-y-2">
          <StatBar label="Entropy Derivative" value={entropy?.entropy_derivative ?? 0} color="warning" size="md" />
          <StatBar label="Signal Entropy" value={entropy?.signal_entropy ?? 0} color="secondary" size="md" />
          <StatBar label="Noise Floor" value={entropy?.noise_floor_estimate ?? 0} color="alert" size="md" />
        </div>
      </Card>
    </div>
  )
}
