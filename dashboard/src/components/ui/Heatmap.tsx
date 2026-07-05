import { useRef, useEffect, useMemo } from 'react'

interface HeatmapCell {
  x: number
  y: number
  value: number
  label?: string
  delta?: number
}

interface HeatmapProps {
  data: HeatmapCell[]
  width?: number
  height?: number
  minColor?: string
  maxColor?: string
  min?: number
  max?: number
  cellSize?: number
  gap?: number
  animate?: boolean
}

function parseColor(hex: string) {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return { r, g, b }
}

function lerpColor(c1: { r: number; g: number; b: number }, c2: { r: number; g: number; b: number }, t: number) {
  return {
    r: Math.round(c1.r + (c2.r - c1.r) * t),
    g: Math.round(c1.g + (c2.g - c1.g) * t),
    b: Math.round(c1.b + (c2.b - c1.b) * t),
  }
}

export default function Heatmap({
  data,
  width = 300,
  height = 200,
  minColor = '#001a0a',
  maxColor = '#00ff88',
  min: minOverride,
  max: maxOverride,
  cellSize = 20,
  gap = 2,
  animate = true,
}: HeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const prevDataRef = useRef<Map<string, number>>(new Map())

  const { min, max, range } = useMemo(() => {
    if (data.length === 0) return { min: 0, max: 1, range: 1 }
    const mn = minOverride ?? Math.min(...data.map((d) => d.value))
    const mx = maxOverride ?? Math.max(...data.map((d) => d.value))
    return { min: mn, max: mx, range: mx - mn || 1 }
  }, [data, minOverride, maxOverride])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || data.length === 0) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = width * dpr
    canvas.height = height * dpr
    ctx.scale(dpr, dpr)

    const minC = parseColor(minColor)
    const maxC = parseColor(maxColor)
    const prevMap = prevDataRef.current

    // Delta-based rendering: only redraw cells that changed
    for (const cell of data) {
      const prev = prevMap.get(`${cell.x},${cell.y}`)
      const changed = prev === undefined || Math.abs(prev - cell.value) > 0.001
      prevMap.set(`${cell.x},${cell.y}`, cell.value)

      if (!changed && !animate) continue

      const t = (cell.value - min) / range
      const color = lerpColor(minC, maxC, Math.min(1, Math.max(0, t)))

      // Animate transition with flash effect
      if (animate && prev !== undefined && changed) {
        const flash = Math.abs(cell.value - prev) > 0.05
        if (flash) {
          ctx.fillStyle = 'rgba(0, 255, 136, 0.3)'
          ctx.fillRect(
            cell.x * (cellSize + gap),
            cell.y * (cellSize + gap),
            cellSize,
            cellSize
          )
        }
      }

      ctx.fillStyle = `rgb(${color.r},${color.g},${color.b})`
      ctx.fillRect(
        cell.x * (cellSize + gap),
        cell.y * (cellSize + gap),
        cellSize,
        cellSize
      )

      // Delta indicator overlay
      if (cell.delta !== undefined && Math.abs(cell.delta) > 0.01) {
        ctx.fillStyle = cell.delta > 0 ? 'rgba(0,255,136,0.2)' : 'rgba(255,68,68,0.2)'
        ctx.fillRect(
          cell.x * (cellSize + gap) + cellSize - 4,
          cell.y * (cellSize + gap),
          4,
          cellSize
        )
      }

      // Label
      if (cell.label) {
        ctx.fillStyle = 'rgba(0,0,0,0.5)'
        ctx.font = '7px monospace'
        ctx.textAlign = 'center'
        ctx.fillText(
          cell.label,
          cell.x * (cellSize + gap) + cellSize / 2,
          cell.y * (cellSize + gap) + cellSize / 2 + 2
        )
      }
    }

    // Grid lines
    ctx.strokeStyle = 'rgba(26, 58, 42, 0.3)'
    ctx.lineWidth = 0.5
    for (const cell of data) {
      ctx.strokeRect(
        cell.x * (cellSize + gap),
        cell.y * (cellSize + gap),
        cellSize,
        cellSize
      )
    }
  }, [data, width, height, minColor, maxColor, min, max, range, cellSize, gap, animate])

  return (
    <canvas
      ref={canvasRef}
      style={{ width, height }}
      className="rounded"
    />
  )
}
