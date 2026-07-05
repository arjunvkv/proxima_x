import { Outlet } from 'react-router-dom'
import Header from './Header'
import Sidebar from './Sidebar'
import StatusBar from './StatusBar'
import ConnectionBanner from './ConnectionBanner'
import StaleIndicator from './StaleIndicator'

export default function DashboardLayout() {
  return (
    <div className="h-screen flex flex-col bg-background terminal-grid">
      <div className="scanline-overlay" />
      <ConnectionBanner />
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <main className="flex-1 overflow-y-auto p-3">
          <Outlet />
        </main>
      </div>
      <StatusBar />
      <StaleIndicator />
    </div>
  )
}
