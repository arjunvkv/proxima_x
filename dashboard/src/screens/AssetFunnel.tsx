import { useMemo } from 'react'
import Card from '../components/ui/Card'
import StatBar from '../components/ui/StatBar'
import VirtualTable from '../components/ui/VirtualTable'
import type { Column } from '../components/ui/VirtualTable'
import { useAssets, useFunnel } from '../core/store/selectors'

interface AssetRow {
  symbol: string
  price: number
  spread: number
  regime: string
  entropy: number
}

const columns: Column<AssetRow>[] = [
  { key: 'symbol', label: 'Symbol', width: 80 },
  { key: 'price', label: 'Price', width: 100, align: 'right' },
  { key: 'spread', label: 'Spread', width: 80, align: 'right' },
  { key: 'regime', label: 'Regime', width: 100 },
  { key: 'entropy', label: 'Entropy', width: 80, align: 'right' },
]

export default function AssetFunnel() {
  const assets = useAssets()
  const funnel = useFunnel()

  const rows: AssetRow[] = useMemo(() => {
    return assets.map((a) => ({
      symbol: a.symbol,
      price: a.price,
      spread: a.spread,
      regime: a.regime,
      entropy: a.entropy,
    }))
  }, [assets])

  return (
    <div className="space-y-3">
      <h2 className="font-headline text-sm text-primary uppercase tracking-widest">
        Asset Evaluation & Signal Funnel
      </h2>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        <Card title="Asset Table" className="lg:col-span-2">
          <VirtualTable
            data={rows}
            columns={columns}
            getRowKey={(r) => r.symbol}
            visibleRows={20}
          />
        </Card>

        <Card title="Signal Funnel">
          <div className="space-y-3">
            <StatBar label="Generated" value={funnel?.generated ?? 0} max={100} color="secondary" size="md" />
            <StatBar label="Threshold Passed" value={funnel?.threshold_passed ?? 0} max={100} color="primary" size="md" />
            <StatBar label="Submitted" value={funnel?.submitted ?? 0} max={100} color="accent" size="md" />
            <StatBar label="Accepted" value={funnel?.accepted ?? 0} max={100} color="tpi" size="md" />
            <StatBar label="Opened" value={funnel?.opened ?? 0} max={100} color="health" size="md" />
            <div className="pt-2 border-t border-border">
              <StatBar label="Leakage" value={funnel?.leakage_pct ?? 0} max={100} color="alert" size="md" />
            </div>
          </div>
        </Card>
      </div>
    </div>
  )
}
