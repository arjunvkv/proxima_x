import { useEffect, type ReactNode } from 'react'
import { getWSClient } from '../../core/websocket/wsClient'

interface Props {
  children: ReactNode
  wsUrl?: string
}

export default function WebSocketProvider({ children, wsUrl }: Props) {
  useEffect(() => {
    const client = getWSClient(wsUrl)
    client.connect()

    return () => {
      client.disconnect()
    }
  }, [wsUrl])

  return <>{children}</>
}
