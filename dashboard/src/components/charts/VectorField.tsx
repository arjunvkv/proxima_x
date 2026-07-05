import { useRef, useEffect } from 'react'

interface VectorFieldProps {
  vector: number[]
  width?: number
  height?: number
  groups?: { label: string; color: string; indices: [number, number] }[]
}

export default function VectorField({
  vector,
  width = 600,
  height = 120,
  groups = [],
}: VectorFieldProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || vector.length === 0) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = width * dpr
    canvas.height = height * dpr
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, width, height)

    const barW = Math.max(4, (width - 20) / vector.length - 1)
    const padding = 10

    // Draw grouped bars
    if (groups.length > 0) {
      for (const group of groups) {
        const [start, end] = group.indices
        for (let i = start; i < end && i < vector.length; i++) {
          const val = Math.min(1, Math.max(0, vector[i] ?? 0))
          const x = padding + i * (barW + 1)
          const barH = val * (height - padding * 2)

          ctx.fillStyle = group.color + '99'
          ctx.fillRect(x, height - padding - barH, barW, barH)
        }
      }
    } else {
      // Flat bars
      for (let i = 0; i < vector.length; i++) {
        const val = Math.min(1, Math.max(0, vector[i] ?? 0))
        const x = padding + i * (barW + 1)
        const barH = val * (height - padding * 2)

        const hue = 120 - val * 120
        ctx.fillStyle = `hsla(${hue}, 100%, 50%, 0.6)`
        ctx.fillRect(x, height - padding - barH, barW, barH)
      }
    }

    // Baseline
    ctx.strokeStyle = '#1a3a2a'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(padding, height - padding)
    ctx.lineTo(width - padding, height - padding)
    ctx.stroke()
  }, [vector, width, height, groups])

  return (
    <canvas
      ref={canvasRef}
      style={{ width, height }}
      className="rounded"
    />
  )
}
