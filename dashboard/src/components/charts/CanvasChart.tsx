import { useRef, useEffect } from 'react'

interface CanvasChartProps {
  width: number
  height: number
  draw: (ctx: CanvasRenderingContext2D, width: number, height: number) => void
  fps?: number
}

export default function CanvasChart({ width, height, draw, fps = 30 }: CanvasChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef<number>(0)
  const lastDrawRef = useRef(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = width * dpr
    canvas.height = height * dpr

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const interval = 1000 / fps

    const loop = (time: number) => {
      const elapsed = time - lastDrawRef.current
      if (elapsed >= interval) {
        ctx.save()
        ctx.scale(dpr, dpr)
        ctx.clearRect(0, 0, width, height)
        draw(ctx, width, height)
        ctx.restore()
        lastDrawRef.current = time
      }
      rafRef.current = requestAnimationFrame(loop)
    }

    rafRef.current = requestAnimationFrame(loop)

    return () => {
      cancelAnimationFrame(rafRef.current)
    }
  }, [width, height, draw, fps])

  return (
    <canvas
      ref={canvasRef}
      style={{ width, height }}
      className="rounded"
    />
  )
}
