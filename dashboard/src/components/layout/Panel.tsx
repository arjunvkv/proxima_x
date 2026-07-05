import { type ReactNode } from 'react'

interface PanelProps {
  title?: string
  children: ReactNode
  className?: string
  rightContent?: ReactNode
  variant?: 'default' | 'glow'
}

export default function Panel({
  title,
  children,
  className = '',
  rightContent,
  variant = 'default',
}: PanelProps) {
  const variantClass = variant === 'glow' ? 'shadow-glow border-primary/20' : ''

  return (
    <div className={`terminal-panel rounded flex flex-col ${variantClass} ${className}`}>
      {title && (
        <div className="flex items-center justify-between px-3 py-2 border-b border-border bg-black/20">
          <h3 className="font-headline text-[10px] text-primary uppercase tracking-widest font-bold">
            {title}
          </h3>
          {rightContent && <div>{rightContent}</div>}
        </div>
      )}
      <div className="flex-1 p-3">
        {children}
      </div>
    </div>
  )
}
