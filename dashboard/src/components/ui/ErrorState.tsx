interface ErrorStateProps {
  message?: string
  onRetry?: () => void
}

export default function ErrorState({
  message = 'An error occurred',
  onRetry,
}: ErrorStateProps) {
  return (
    <div className="terminal-panel rounded p-6 flex flex-col items-center justify-center gap-3">
      <span className="material-symbols-outlined text-alert text-3xl">error</span>
      <span className="text-alert font-data text-sm">{message}</span>
      {onRetry && (
        <button
          onClick={onRetry}
          className="px-4 py-1.5 bg-alert/10 border border-alert/30 rounded text-alert text-xs font-data uppercase tracking-wider hover:bg-alert/20 transition-colors"
        >
          Retry
        </button>
      )}
    </div>
  )
}
