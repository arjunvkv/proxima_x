export const stitchTokens = {
  colors: {
    primary: '#00ff88',
    secondary: '#00ccff',
    accent: '#aa66ff',
    alert: '#ff4444',
    warning: '#ffdd00',
    background: '#0a0a0a',
    surface: '#0d0d0d',
    border: '#1a3a2a',
    regime: '#ff8800',
    thermo: '#ff00ff',
    tpi: '#00ffff',
    exec: '#0088ff',
    shadow: '#aa66ff',
    health: '#00ff88',
  },
  fonts: {
    headline: "'DM Sans', sans-serif",
    display: "'DM Sans', sans-serif",
    body: "'IBM Plex Sans', sans-serif",
    data: "'Courier Prime', monospace",
    label: "'Public Sans', sans-serif",
  },
  shadows: {
    glow: '0 0 12px rgba(0,255,136,0.4)',
    glowBlue: '0 0 12px rgba(0,204,255,0.4)',
    glowBorder: 'inset 0 0 10px rgba(0,255,136,0.1), 0 0 5px rgba(0,255,136,0.1)',
  },
  animations: {
    scanline: 'scanline 6s linear infinite',
    pulseSlow: 'pulse-slow 3s ease-in-out infinite',
    ping: 'ping 2s cubic-bezier(0, 0, 0.2, 1) infinite',
  },
  spacing: {
    card: '8px',
    section: '12px',
    grid: '6px',
  },
  breakpoints: {
    desktop: '1280px',
    tablet: '768px',
  },
} as const

export type StitchToken = typeof stitchTokens
