import type { WSConnectionState, WSConnectionEvent } from '../../types/ws'

export type WSStateListener = (event: WSConnectionEvent) => void

export class WSStateMachine {
  private _state: WSConnectionState = 'DISCONNECTED'
  private _attempts = 0
  private _listeners: Set<WSStateListener> = new Set()

  get state(): WSConnectionState {
    return this._state
  }

  get attempts(): number {
    return this._attempts
  }

  subscribe(listener: WSStateListener): () => void {
    this._listeners.add(listener)
    return () => this._listeners.delete(listener)
  }

  private emit(reason?: string) {
    const event: WSConnectionEvent = {
      state: this._state,
      timestamp: Date.now(),
      attempt: this._attempts,
      reason,
    }
    this._listeners.forEach((fn) => fn(event))
  }

  transitioning() {
    this._state = 'CONNECTING'
    this._attempts++
    this.emit()
  }

  connected() {
    this._state = 'CONNECTED'
    this._attempts = 0
    this.emit()
  }

  streaming() {
    this._state = 'STREAMING'
    this.emit()
  }

  reconnecting(reason?: string) {
    this._state = 'RECONNECTING'
    this.emit(reason)
  }

  error(reason?: string) {
    this._state = 'ERROR'
    this.emit(reason)
  }

  disconnected(reason?: string) {
    this._state = 'DISCONNECTED'
    this.emit(reason)
  }

  reset() {
    this._attempts = 0
    this._state = 'DISCONNECTED'
  }
}
