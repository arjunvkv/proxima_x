import { useState, useEffect } from 'react'

interface FpsState {
  fps: number
  maxFps: number
  minFps: number
  frameTimes: number[]
}

export default function FpsMonitor() {
  const [fpsState, setFpsState] = useState<FpsState>({ fps: 0, maxFps: 0, minFps: 999, frameTimes: [] })

  useEffect(() => {
    if (import.meta.env.PROD) return

    let lastTime = performance.now()
    let frameCount = 0
    let rafId = 0

    const tick = () => {
      frameCount++
      const now = performance.now()
      if (now - lastTime >= 1000) {
        const currentFps = Math.round((frameCount * 1000) / (now - lastTime))
        setFpsState((prev) => ({
          fps: currentFps,
          maxFps: Math.max(prev.maxFps, currentFps),
          minFps: Math.min(prev.minFps, currentFps),
          frameTimes: [...prev.frameTimes.slice(-59), currentFps],
        }))
        frameCount = 0
        lastTime = now
      }
      rafId = requestAnimationFrame(tick)
    }

    rafId = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafId)
  }, [])

  if (import.meta.env.PROD) return null

  const color = fpsState.fps >= 55 ? 'text-primary' : fpsState.fps >= 30 ? 'text-warning' : 'text-alert'

  return (
    <div className="fixed top-14 right-2 z-50 bg-black/80 border border-border rounded px-2 py-1 font-data text-[9px] leading-relaxed">
      <div className={`${color}`}>FPS: {fpsState.fps}</div>
      <div className="text-gray-500">min: {fpsState.minFps} max: {fpsState.maxFps}</div>
    </div>
  )
}
