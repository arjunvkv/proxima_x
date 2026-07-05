import { type ReactNode } from 'react'

interface CardProps {
  title?: string
  children: ReactNode
  glow?: 'green' | 'blue' | 'none'
  className?: string
  titleRight?: ReactNode
}

export default function Card({ title, children, glow = 'none', className = '', titleRight }: CardProps) {
  const glowClass =
    glow === 'green' ? 'shadow-glow border-primary/30' :
    glow === 'blue' ? 'shadow-glow-blue border-secondary/30' :
    'border-border'

  return (
    <div className={`terminal-panel rounded p-3 flex flex-col ${glowClass} ${className}`}>
      {title && (
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-headline text-xs text-primary uppercase tracking-widest">
            {title}
          </h3>
          {titleRight && <div>{titleRight}</div>}
        </div>
      )}
      {children}
    </div>
  )
}
