interface GaugeProps {
  value: number
  label: string
  min?: number
  max?: number
  threshold?: number
  color?: string
  size?: 'sm' | 'md' | 'lg'
}

const sizeMap = {
  sm: { container: 'w-16 h-16', text: 'text-[10px]', value: 'text-sm' },
  md: { container: 'w-24 h-24', text: 'text-xs', value: 'text-lg' },
  lg: { container: 'w-32 h-32', text: 'text-sm', value: 'text-2xl' },
}

export default function Gauge({
  value,
  label,
  min = 0,
  max = 1,
  threshold,
  color = '#00ff88',
  size = 'md',
}: GaugeProps) {
  const pct = Math.min(100, Math.max(0, ((value - min) / (max - min)) * 100))
  const circumference = 2 * Math.PI * 40
  const offset = circumference - (pct / 100) * circumference
  const s = sizeMap[size]

  const thresholdColor = threshold !== undefined && value < threshold ? '#ff4444' : color

  return (
    <div className={`flex flex-col items-center gap-1 ${s.container}`}>
      <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
        <circle
          cx="50"
          cy="50"
          r="40"
          fill="none"
          stroke="#1a3a2a"
          strokeWidth="6"
        />
        <circle
          cx="50"
          cy="50"
          r="40"
          fill="none"
          stroke={thresholdColor}
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-500"
        />
      </svg>
      <div className="absolute flex flex-col items-center justify-center">
        <span className={`${s.value} font-data font-bold`} style={{ color: thresholdColor }}>
          {value.toFixed(2)}
        </span>
      </div>
      <span className={`${s.text} text-gray-400 font-data uppercase text-center`}>
        {label}
      </span>
    </div>
  )
}
