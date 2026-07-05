import Card from '../components/ui/Card'
import Gauge from '../components/ui/Gauge'
import StatBar from '../components/ui/StatBar'
import { useDashboardEntropy } from '../core/store/selectors'

export default function Thermodynamics() {
  const entropy = useDashboardEntropy()

  return (
    <div className="space-y-3">
      <h2 className="font-headline text-sm text-primary uppercase tracking-widest">
        Tick Thermodynamics Gauges
      </h2>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-3">
        <Card title="Signal Energy" glow="green">
          <div className="flex justify-center py-4">
            <Gauge value={entropy?.signal_entropy ?? 0} label="Signal" color="#00ff88" size="lg" />
          </div>
        </Card>

        <Card title="Noise Floor" glow="blue">
          <div className="flex justify-center py-4">
            <Gauge value={entropy?.noise_floor_estimate ?? 0} label="Noise" color="#00ccff" size="lg" />
          </div>
        </Card>

        <Card title="Entropy Derivative">
          <div className="flex justify-center py-4">
            <Gauge value={entropy?.entropy_derivative ?? 0} label="Derivative" color="#ff8800" size="lg" />
          </div>
        </Card>

        <Card title="Predictability">
          <div className="flex justify-center py-4">
            <Gauge value={entropy?.predictability_index ?? 0} label="Predict" color="#aa66ff" size="lg" />
          </div>
        </Card>
      </div>

      <Card title="Thermodynamics Detail">
        <div className="space-y-2">
          <StatBar label="Entropy Level" value={entropy?.entropy_level ?? 0} color="thermo" size="md" />
          <StatBar label="Compression Ratio" value={entropy?.compression_ratio ?? 0} color="secondary" size="md" />
          <StatBar label="Noise Floor Estimate" value={entropy?.noise_floor_estimate ?? 0} color="warning" size="md" />
        </div>
      </Card>
    </div>
  )
}
