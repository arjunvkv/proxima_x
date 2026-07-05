import { parseDashboard } from './parserEngine'

self.onmessage = (e: MessageEvent<string>) => {
  try {
    const raw = e.data
    const parsed = parseDashboard(raw)
    self.postMessage(parsed)
  } catch (err) {
    self.postMessage({ error: String(err) })
  }
}
