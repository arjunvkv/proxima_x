import { useRealtimeStore } from '../../core/store/useRealtimeStore'

export class ShaderEngine {
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D | null = null
  private rafId = 0
  private time = 0
  private particles: { x: number; y: number; vx: number; vy: number; size: number; hue: number }[] = []
  private running = false

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas
  }

  start() {
    this.running = true
    this.ctx = this.canvas.getContext('2d')
    if (!this.ctx) return

    const dpr = window.devicePixelRatio || 1
    this.canvas.width = window.innerWidth * dpr
    this.canvas.height = window.innerHeight * dpr
    this.canvas.style.width = window.innerWidth + 'px'
    this.canvas.style.height = window.innerHeight + 'px'

    // Create particles
    this.particles = Array.from({ length: 64 }, () => ({
      x: Math.random() * this.canvas.width,
      y: Math.random() * this.canvas.height,
      vx: (Math.random() - 0.5) * 2,
      vy: (Math.random() - 0.5) * 2,
      size: 1 + Math.random() * 3,
      hue: 120 + Math.random() * 60,
    }))

    window.addEventListener('resize', this.onResize)
    this.loop()
  }

  stop() {
    this.running = false
    cancelAnimationFrame(this.rafId)
    window.removeEventListener('resize', this.onResize)
  }

  private onResize = () => {
    const dpr = window.devicePixelRatio || 1
    this.canvas.width = window.innerWidth * dpr
    this.canvas.height = window.innerHeight * dpr
  }

  private loop = () => {
    if (!this.running || !this.ctx) return
    this.time += 0.016

    const state = useRealtimeStore.getState()
    const vec = state.engineVector
    const integrity = state.systemIntegrity
    const pressure = state.killSwitchPressure
    const avgEnergy = vec.reduce((a, b) => a + b, 0) / Math.max(vec.length, 1)

    const ctx = this.ctx
    const w = this.canvas.width
    const h = this.canvas.height

    // Fade trail
    ctx.fillStyle = `rgba(5, 5, 5, ${0.15 + pressure * 0.1})`
    ctx.fillRect(0, 0, w, h)

    // Update and draw particles
    for (const p of this.particles) {
      // Engine vector drives particle velocity (use modulo for mapping)
      const vi = Math.floor(Math.random() * vec.length)
      const v = vec[vi] ?? 0

      p.vx += (v - 0.5) * 0.1 * (1 + pressure)
      p.vy += (v - 0.5) * 0.1 * (1 + pressure)

      // Damping
      p.vx *= 0.99
      p.vy *= 0.99

      p.x += p.vx
      p.y += p.vy

      // Wrap around edges
      if (p.x < 0) p.x = w
      if (p.x > w) p.x = 0
      if (p.y < 0) p.y = h
      if (p.y > h) p.y = 0

      // Color: green base, shift to red under pressure
      const hue = p.hue - pressure * 60
      const alpha = 0.3 + integrity * 0.5 + avgEnergy * 0.2
      const size = p.size + avgEnergy * 2

      ctx.beginPath()
      ctx.arc(p.x, p.y, size, 0, Math.PI * 2)
      ctx.fillStyle = `hsla(${hue}, 100%, 50%, ${alpha})`
      ctx.fill()

      // Glow
      if (integrity > 0.5) {
        ctx.beginPath()
        ctx.arc(p.x, p.y, size * 2, 0, Math.PI * 2)
        ctx.fillStyle = `hsla(${hue}, 100%, 50%, ${alpha * 0.2})`
        ctx.fill()
      }
    }

    // Center energy pulse
    const centerGlow = ctx.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, 100 + avgEnergy * 200)
    centerGlow.addColorStop(0, `rgba(0, 255, 136, ${0.08 + integrity * 0.15})`)
    centerGlow.addColorStop(1, 'rgba(0, 255, 136, 0)')
    ctx.fillStyle = centerGlow
    ctx.fillRect(0, 0, w, h)

    this.rafId = requestAnimationFrame(this.loop)
  }
}
