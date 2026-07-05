import type { EngineVectorGroup } from '../../types/engine'

interface RenderConfig {
  width: number
  height: number
  vector: number[]
  groups: EngineVectorGroup[]
}

export class VectorRenderer {
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D | null = null

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas
    this.ctx = canvas.getContext('2d')
  }

  render(config: RenderConfig) {
    const ctx = this.ctx
    if (!ctx) return

    const { width, height, vector, groups } = config
    const dpr = window.devicePixelRatio || 1
    this.canvas.width = width * dpr
    this.canvas.height = height * dpr
    ctx.scale(dpr, dpr)

    ctx.clearRect(0, 0, width, height)

    const padding = { top: 10, bottom: 20, left: 10, right: 10 }
    const plotW = width - padding.left - padding.right
    const plotH = height - padding.top - padding.bottom
    const barGap = 1

    // Grouped colored bars
    for (const group of groups) {
      const [start, end] = group.indices
      const groupSize = end - start
      const totalBars = vector.length
      const barW = (plotW - (totalBars - 1) * barGap) / totalBars

      ctx.fillStyle = group.color + '99'

      for (let i = start; i < end && i < vector.length; i++) {
        const val = Math.min(1, Math.max(0, vector[i] ?? 0))
        const x = padding.left + i * (barW + barGap)
        const barH = val * plotH
        const y = height - padding.bottom - barH

        ctx.fillRect(x, y, barW, barH)

        // Glow highlight for high values
        if (val > 0.7) {
          ctx.fillStyle = group.color + '44'
          ctx.fillRect(x - 1, y - 1, barW + 2, barH + 2)
          ctx.fillStyle = group.color + '99'
        }
      }
    }

    // Baseline
    ctx.strokeStyle = '#1a3a2a'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(padding.left, height - padding.bottom)
    ctx.lineTo(width - padding.right, height - padding.bottom)
    ctx.stroke()
  }
}
