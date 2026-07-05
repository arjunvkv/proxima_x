import { type ReactNode } from 'react'
import ErrorBoundary from './ErrorBoundary'

interface Props {
  screenName: string
  children: ReactNode
}

export default function ScreenErrorBoundary({ screenName, children }: Props) {
  return (
    <ErrorBoundary
      onError={(error) => {
        console.error(`[${screenName}] Screen crashed:`, error)
      }}
    >
      {children}
    </ErrorBoundary>
  )
}
