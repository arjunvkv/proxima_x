import { create } from 'zustand'
import type { WSConnectionState } from '../../types/ws'

export interface UIState {
  wsStatus: WSConnectionState
  wsAttempts: number
  wsLastError: string | null
  lastLiveTs: number
  lastStateTs: number
  lastDashboardTs: number
  sidebarCollapsed: boolean
  activeScreen: string
}

export interface UIActions {
  setWsStatus: (status: WSConnectionState, attempt?: number, error?: string) => void
  updateTimestamp: (stream: 'live' | 'state' | 'dashboard') => void
  toggleSidebar: () => void
  setActiveScreen: (screen: string) => void
  reset: () => void
}

function isStale(ts: number, threshold = 3000): boolean {
  return Date.now() - ts > threshold
}

const initialState: UIState = {
  wsStatus: 'DISCONNECTED',
  wsAttempts: 0,
  wsLastError: null,
  lastLiveTs: 0,
  lastStateTs: 0,
  lastDashboardTs: 0,
  sidebarCollapsed: false,
  activeScreen: 'dashboard',
}

export const useUIStateStore = create<UIState & UIActions>((set) => ({
  ...initialState,

  setWsStatus: (status, attempt, error) =>
    set({
      wsStatus: status,
      wsAttempts: attempt ?? 0,
      wsLastError: error ?? null,
    }),

  updateTimestamp: (stream) =>
    set({ [`last${stream.charAt(0).toUpperCase() + stream.slice(1)}Ts`]: Date.now() } as Partial<UIState>),

  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),

  setActiveScreen: (screen) => set({ activeScreen: screen }),

  reset: () => set({ ...initialState }),
}))

export { isStale }
