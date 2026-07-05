import { useRef, useEffect } from 'react'

interface SparklineProps {
  data: number[]
  width?: number
  height?: number
  color?: string
  fillColor?: string
  max?: number
  min?: number
}

export default function Sparkline({
  data,
  width = 120,
  height = 32,
  color = '#00ff88',
  fillColor = 'rgba(0,255,136,0.15)',
  max: maxOverride,
  min: minOverride,
}: SparklineProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || data.length < 2) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = width * dpr
    canvas.height = height * dpr
    ctx.scale(dpr, dpr)

    const min = minOverride ?? Math.min(...data)
    const max = maxOverride ?? Math.max(...data)
    const range = max - min || 1
    const padding = 2

    const plotW = width - padding * 2
    const plotH = height - padding * 2

    ctx.clearRect(0, 0, width, height)

    // Fill area
    ctx.beginPath()
    data.forEach((val, i) => {
      const x = padding + (i / (data.length - 1)) * plotW
      const y = padding + plotH - ((val - min) / range) * plotH
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
    })
    ctx.lineTo(padding + plotW, height - padding)
    ctx.lineTo(padding, height - padding)
    ctx.closePath()
    ctx.fillStyle = fillColor
    ctx.fill()

    // Draw line
    ctx.beginPath()
    data.forEach((val, i) => {
      const x = padding + (i / (data.length - 1)) * plotW
      const y = padding + plotH - ((val - min) / range) * plotH
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
    })
    ctx.strokeStyle = color
    ctx.lineWidth = 1.5
    ctx.stroke()
  }, [data, width, height, color, fillColor, maxOverride, minOverride])

  return (
    <canvas
      ref={canvasRef}
      style={{ width, height }}
      className="inline-block"
    />
  )
}
