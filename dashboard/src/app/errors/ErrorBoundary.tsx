import { Component, type ReactNode, type ErrorInfo } from 'react'

interface Props {
  children: ReactNode
  fallback?: ReactNode
  onError?: (error: Error, errorInfo: ErrorInfo) => void
}

interface State {
  hasError: boolean
  error: Error | null
}

export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('[ErrorBoundary]', error, errorInfo)
    this.props.onError?.(error, errorInfo)
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null })
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      return (
        <div className="terminal-panel rounded p-6 flex flex-col items-center justify-center gap-3 m-3">
          <span className="material-symbols-outlined text-alert text-3xl">warning</span>
          <span className="text-alert font-data text-sm uppercase tracking-wider">
            System Module Failed
          </span>
          <span className="text-gray-500 text-[10px] font-data max-w-md text-center">
            {this.state.error?.message ?? 'Unknown error'}
          </span>
          <button
            onClick={this.handleRetry}
            className="px-4 py-1.5 bg-alert/10 border border-alert/30 rounded text-alert text-xs font-data uppercase tracking-wider hover:bg-alert/20 transition-colors"
          >
            Retry Module
          </button>
        </div>
      )
    }

    return this.props.children
  }
}
