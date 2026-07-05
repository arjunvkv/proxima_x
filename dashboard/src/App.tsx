import { RouterProvider } from 'react-router-dom'
import { router } from './app/router'
import QueryProvider from './app/providers/QueryProvider'
import WebSocketProvider from './app/providers/WebSocketProvider'
import BootLoader from './app/layout/BootLoader'
import FpsMonitor from './app/dev/FpsMonitor'

export default function App() {
  return (
    <QueryProvider>
      <WebSocketProvider>
        <BootLoader>
          <RouterProvider router={router} />
        </BootLoader>
        <FpsMonitor />
      </WebSocketProvider>
    </QueryProvider>
  )
}
