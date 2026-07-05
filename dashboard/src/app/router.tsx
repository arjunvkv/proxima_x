import { lazy, Suspense } from 'react'
import { createBrowserRouter } from 'react-router-dom'
import DashboardLayout from './layout/DashboardLayout'
import ScreenErrorBoundary from './errors/ScreenErrorBoundary'

// Lazy-loaded screens
const MainDashboard = lazy(() => import('../screens/MainDashboard'))
const AssetFunnel = lazy(() => import('../screens/AssetFunnel'))
const Intelligence = lazy(() => import('../screens/Intelligence'))
const Entropy = lazy(() => import('../screens/Entropy'))
const Thermodynamics = lazy(() => import('../screens/Thermodynamics'))
const TPI = lazy(() => import('../screens/TPI'))
const ResearchDirector = lazy(() => import('../screens/ResearchDirector'))
const RiskShadow = lazy(() => import('../screens/RiskShadow'))
const ShaderCanvas = lazy(() => import('../screens/ShaderCanvas'))

function ScreenLoader() {
  return (
    <div className="flex items-center justify-center h-full">
      <div className="flex flex-col items-center gap-3">
        <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin" />
        <span className="text-xs text-gray-500 font-data uppercase">Loading...</span>
      </div>
    </div>
  )
}

export const router = createBrowserRouter([
  {
    path: '/',
    element: <DashboardLayout />,
    children: [
      {
        index: true,
        element: (
          <ScreenErrorBoundary screenName="MainDashboard">
            <Suspense fallback={<ScreenLoader />}>
              <MainDashboard />
            </Suspense>
          </ScreenErrorBoundary>
        ),
      },
      {
        path: 'assets',
        element: (
          <ScreenErrorBoundary screenName="AssetFunnel">
            <Suspense fallback={<ScreenLoader />}>
              <AssetFunnel />
            </Suspense>
          </ScreenErrorBoundary>
        ),
      },
      {
        path: 'intelligence',
        element: (
          <ScreenErrorBoundary screenName="Intelligence">
            <Suspense fallback={<ScreenLoader />}>
              <Intelligence />
            </Suspense>
          </ScreenErrorBoundary>
        ),
      },
      {
        path: 'entropy',
        element: (
          <ScreenErrorBoundary screenName="Entropy">
            <Suspense fallback={<ScreenLoader />}>
              <Entropy />
            </Suspense>
          </ScreenErrorBoundary>
        ),
      },
      {
        path: 'thermo',
        element: (
          <ScreenErrorBoundary screenName="Thermodynamics">
            <Suspense fallback={<ScreenLoader />}>
              <Thermodynamics />
            </Suspense>
          </ScreenErrorBoundary>
        ),
      },
      {
        path: 'tpi',
        element: (
          <ScreenErrorBoundary screenName="TPI">
            <Suspense fallback={<ScreenLoader />}>
              <TPI />
            </Suspense>
          </ScreenErrorBoundary>
        ),
      },
      {
        path: 'research',
        element: (
          <ScreenErrorBoundary screenName="ResearchDirector">
            <Suspense fallback={<ScreenLoader />}>
              <ResearchDirector />
            </Suspense>
          </ScreenErrorBoundary>
        ),
      },
      {
        path: 'risk',
        element: (
          <ScreenErrorBoundary screenName="RiskShadow">
            <Suspense fallback={<ScreenLoader />}>
              <RiskShadow />
            </Suspense>
          </ScreenErrorBoundary>
        ),
      },
      {
        path: 'shader',
        element: (
          <ScreenErrorBoundary screenName="ShaderCanvas">
            <Suspense fallback={<ScreenLoader />}>
              <ShaderCanvas />
            </Suspense>
          </ScreenErrorBoundary>
        ),
      },
    ],
  },
])
