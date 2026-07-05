/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
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
      fontFamily: {
        headline: ['DM Sans', 'sans-serif'],
        display: ['DM Sans', 'sans-serif'],
        body: ['IBM Plex Sans', 'sans-serif'],
        data: ['Courier Prime', 'monospace'],
        label: ['Public Sans', 'sans-serif'],
      },
      borderRadius: {
        DEFAULT: '0.125rem',
        lg: '0.25rem',
        xl: '0.5rem',
        full: '0.75rem',
      },
      boxShadow: {
        glow: '0 0 12px rgba(0,255,136,0.4)',
        'glow-blue': '0 0 12px rgba(0,204,255,0.4)',
        'glow-border': 'inset 0 0 10px rgba(0,255,136,0.1), 0 0 5px rgba(0,255,136,0.1)',
      },
      backgroundImage: {
        grid: 'linear-gradient(rgba(0,255,136,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,255,136,0.03) 1px, transparent 1px)',
      },
      backgroundSize: {
        grid: '20px 20px',
      },
      keyframes: {
        scanline: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100%)' },
        },
        'pulse-slow': {
          '0%, 100%': { opacity: '0.6' },
          '50%': { opacity: '1' },
        },
      },
      animation: {
        scanline: 'scanline 6s linear infinite',
        'pulse-slow': 'pulse-slow 3s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
