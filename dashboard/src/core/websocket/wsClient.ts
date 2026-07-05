import { WSStateMachine } from './wsStateMachine'
import { streamRouter } from './streamRouter'
import { useUIStateStore } from '../store/useUIStateStore'
import { resetBatching } from '../../utils/batching'

const RECONNECT_BASE = 500
const RECONNECT_MAX = 8000
const HEARTBEAT_INTERVAL = 5000
const STALE_THRESHOLD = 5000

export class WSClient {
  private _ws: WebSocket | null = null
  private _url: string
  private _fsm: WSStateMachine
  private _reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private _heartbeatTimer: ReturnType<typeof setInterval> | null = null
  private _staleTimer: ReturnType<typeof setInterval> | null = null
  private _destroyed = false

  constructor(url = 'ws://localhost:8765') {
    this._url = url
    this._fsm = new WSStateMachine()

    this._fsm.subscribe((event) => {
      useUIStateStore.getState().setWsStatus(
        event.state,
        event.attempt,
        event.reason
      )
    })
  }

  connect() {
    this._destroyed = false
    if (this._ws?.readyState === WebSocket.OPEN) return

    console.log('[WS] Connecting to', this._url)
    this._fsm.transitioning()

    try {
      this._ws = new WebSocket(this._url)
    } catch (err) {
      console.error('[WS] Connection error:', err)
      this._fsm.error(String(err))
      this._scheduleReconnect()
      return
    }

    this._ws.onopen = () => {
      if (this._destroyed) return
      console.log('[WS] Connected')
      this._fsm.connected()
      this._startHeartbeat()
    }

    this._ws.onmessage = (event: MessageEvent) => {
      if (this._destroyed) return
      try {
        const data = JSON.parse(event.data)
        console.log('[WS] Received:', data.stream ?? 'unknown')
        streamRouter(data)
      } catch {
        console.warn('[WS] Parse error on message')
      }
    }

    this._ws.onclose = (event) => {
      this._stopHeartbeat()
      if (this._destroyed) return
      const reason = event.code === 1000 ? undefined : `WS closed (code ${event.code})`
      console.warn('[WS] Closed:', reason || 'clean')
      this._fsm.reconnecting(reason)
      this._scheduleReconnect()
    }

    this._ws.onerror = () => {
      if (this._destroyed) return
      console.error('[WS] Error event')
      this._fsm.error('WS error')
    }
  }

  disconnect() {
    this._destroyed = true
    this._stopHeartbeat()
    this._cancelReconnect()

    if (this._ws) {
      this._ws.onopen = null
      this._ws.onmessage = null
      this._ws.onclose = null
      this._ws.onerror = null
      if (this._ws.readyState === WebSocket.OPEN) {
        this._ws.close(1000, 'Client disconnect')
      }
      this._ws = null
    }

    this._fsm.disconnected('Client disconnect')
    resetBatching()
  }

  get state() {
    return this._fsm.state
  }

  private _scheduleReconnect() {
    if (this._destroyed) return
    this._cancelReconnect()

    const delay = Math.min(
      RECONNECT_BASE * Math.pow(2, this._fsm.attempts) + Math.random() * 300,
      RECONNECT_MAX
    )

    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null
      this.connect()
    }, delay)
  }

  private _cancelReconnect() {
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer)
      this._reconnectTimer = null
    }
  }

  private _startHeartbeat() {
    this._stopHeartbeat()
    this._heartbeatTimer = setInterval(() => {
      if (this._ws?.readyState === WebSocket.OPEN) {
        this._ws.send(JSON.stringify({ type: 'ping' }))
      }
    }, HEARTBEAT_INTERVAL)
  }

  private _stopHeartbeat() {
    if (this._heartbeatTimer) {
      clearInterval(this._heartbeatTimer)
      this._heartbeatTimer = null
    }
    if (this._staleTimer) {
      clearInterval(this._staleTimer)
      this._staleTimer = null
    }
  }
}

// Singleton
let _instance: WSClient | null = null

export function getWSClient(url?: string): WSClient {
  if (!_instance) {
    _instance = new WSClient(url)
  }
  return _instance
}
