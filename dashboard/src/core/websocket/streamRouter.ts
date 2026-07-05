import { pushFrame } from '../../utils/batching'
import { useRealtimeStore } from '../store/useRealtimeStore'
import { useSnapshotStore } from '../store/useSnapshotStore'
import { useDashboardStore } from '../store/useDashboardStore'
import { useUIStateStore } from '../store/useUIStateStore'
import type { WSBaseMessage, LiveFramePayload, TelemetrySnapshot } from '../../types/ws'
import type { ParsedDashboard } from '../../types/domain'

// We'll create the worker lazily
let _worker: Worker | null = null

function getWorker(): Worker {
  if (!_worker) {
    _worker = new Worker(
      new URL('../workers/dashboardParser.worker.ts', import.meta.url),
      { type: 'module' }
    )
    _worker.onmessage = (e: MessageEvent<ParsedDashboard>) => {
      const data = e.data as ParsedDashboard
      if ((data as any).error) {
        console.warn('[StreamRouter] Worker parse error:', (data as any).error)
        return
      }
      useDashboardStore.getState().setParsed(data)
    }
  }
  return _worker
}

export function streamRouter(raw: WSBaseMessage) {
  const stream = raw.stream
  const uiStore = useUIStateStore.getState()

  // Track timestamps
  if (stream === 'live' || stream === 'state' || stream === 'dashboard') {
    uiStore.updateTimestamp(stream)
  }

  switch (stream) {
    case 'live': {
      const msg = raw as unknown as { stream: 'live'; type: string } & LiveFramePayload
      const frame: LiveFramePayload = {
        engine_vector: (msg as any).engine_vector ?? [],
        system_integrity: (msg as any).system_integrity ?? 0,
        kill_switch_pressure: (msg as any).kill_switch_pressure ?? 0,
        execution_intensity: (msg as any).execution_intensity ?? 0,
        rollout_progress: (msg as any).rollout_progress ?? 0,
        regime_state: (msg as any).regime_state ?? 'IDLE',
        regime_confidence: (msg as any).regime_confidence ?? 0,
        risk_exposure: (msg as any).risk_exposure ?? 0,
        drawdown: (msg as any).drawdown ?? 0,
        symbols: (msg as any).symbols ?? {},
        entropy_global: (msg as any).entropy_global ?? 0,
        shadow_alignment: (msg as any).shadow_alignment ?? 0,
        sof_score: (msg as any).sof_score ?? 0,
        tpi_global: (msg as any).tpi_global ?? 0,
        tpi_alignment_map: (msg as any).tpi_alignment_map ?? {},
        frame_id: (msg as any).frame_id ?? 0,
        timestamp: (msg as any).timestamp ?? Date.now(),
        cycle_count: (msg as any).cycle_count ?? 0,
        alignment: (msg as any).alignment ?? 0,
        stability: (msg as any).stability ?? 0,
        entropy: (msg as any).entropy ?? 0,
        tpi_confidence: (msg as any).tpi_confidence ?? 0,
      }

      pushFrame(frame)

      if (uiStore.wsStatus === 'CONNECTED') {
        useUIStateStore.getState().setWsStatus('STREAMING')
      }
      break
    }

    case 'state': {
      const msg = raw as unknown as { stream: 'state'; type: string } & TelemetrySnapshot
      useSnapshotStore.getState().updateSnapshot(msg)
      break
    }

    case 'dashboard': {
      const msg = raw as unknown as { stream: 'dashboard'; type: string; text: string }
      if (msg.text) {
        getWorker().postMessage(msg.text)
      }
      break
    }

    default:
      break
  }
}
