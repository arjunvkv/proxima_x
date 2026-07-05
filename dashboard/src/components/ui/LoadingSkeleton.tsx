interface LoadingSkeletonProps {
  lines?: number
  variant?: 'card' | 'text' | 'chart'
}

export default function LoadingSkeleton({ lines = 3, variant = 'text' }: LoadingSkeletonProps) {
  if (variant === 'card') {
    return (
      <div className="terminal-panel rounded p-4 space-y-3 animate-pulse">
        <div className="h-3 bg-gray-800 rounded w-1/3" />
        <div className="h-8 bg-gray-800 rounded" />
        <div className="h-8 bg-gray-800 rounded w-2/3" />
      </div>
    )
  }

  if (variant === 'chart') {
    return (
      <div className="terminal-panel rounded p-4 animate-pulse">
        <div className="h-3 bg-gray-800 rounded w-1/4 mb-4" />
        <div className="h-32 bg-gray-800 rounded" />
      </div>
    )
  }

  return (
    <div className="space-y-2 animate-pulse">
      {Array.from({ length: lines }).map((_, i) => (
        <div
          key={i}
          className="h-3 bg-gray-800 rounded"
          style={{ width: `${60 + Math.random() * 40}%` }}
        />
      ))}
    </div>
  )
}
