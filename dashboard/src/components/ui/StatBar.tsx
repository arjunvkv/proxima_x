interface StatBarProps {
  label: string
  value: number
  max?: number
  color?: 'primary' | 'secondary' | 'accent' | 'alert' | 'warning' | 'regime' | 'thermo' | 'tpi' | 'exec' | 'shadow' | 'health'
  showValue?: boolean
  size?: 'sm' | 'md' | 'lg'
}

const colorMap: Record<string, string> = {
  primary: 'bg-primary',
  secondary: 'bg-secondary',
  accent: 'bg-accent',
  alert: 'bg-alert',
  warning: 'bg-warning',
  regime: 'bg-regime',
  thermo: 'bg-thermo',
  tpi: 'bg-tpi',
  exec: 'bg-exec',
  shadow: 'bg-shadow',
  health: 'bg-health',
}

const sizeMap = {
  sm: { bar: 'h-1', text: 'text-[9px]' },
  md: { bar: 'h-1.5', text: 'text-[10px]' },
  lg: { bar: 'h-2', text: 'text-xs' },
}

export default function StatBar({
  label,
  value,
  max = 1,
  color = 'primary',
  showValue = true,
  size = 'sm',
}: StatBarProps) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100))
  const s = sizeMap[size]
  const barColor = colorMap[color] || 'bg-primary'

  return (
    <div className="flex flex-col gap-1">
      <div className={`flex justify-between ${s.text} font-data`}>
        <span className="text-gray-400 uppercase truncate">{label}</span>
        {showValue && (
          <span className="text-primary font-bold">{value.toFixed(3)}</span>
        )}
      </div>
      <div className={`${s.bar} bg-primary/5 w-full overflow-hidden border border-primary/10 rounded`}>
        <div
          className={`h-full ${barColor} shadow-[0_0_8px_rgba(0,255,136,0.5)] transition-all duration-300`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
