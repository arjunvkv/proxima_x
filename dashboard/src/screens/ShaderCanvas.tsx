import { useRef, useEffect } from 'react'
import { ShaderEngine } from '../visualization/webgl/ShaderEngine'

export default function ShaderCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<ShaderEngine | null>(null)

  useEffect(() => {
    if (!canvasRef.current) return
    const engine = new ShaderEngine(canvasRef.current)
    engineRef.current = engine
    engine.start()

    return () => {
      engine.stop()
      engineRef.current = null
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 w-full h-full"
      style={{ zIndex: 0 }}
    />
  )
}
