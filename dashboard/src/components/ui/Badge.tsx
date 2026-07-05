import { type ReactNode } from 'react'

interface BadgeProps {
  children: ReactNode
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info' | 'purple'
  className?: string
}

const variantMap = {
  default: 'bg-gray-800 text-gray-300 border-gray-700',
  success: 'bg-primary/10 text-primary border-primary/20',
  warning: 'bg-warning/10 text-warning border-warning/20',
  danger: 'bg-alert/10 text-alert border-alert/20',
  info: 'bg-secondary/10 text-secondary border-secondary/20',
  purple: 'bg-accent/10 text-accent border-accent/20',
}

export default function Badge({ children, variant = 'default', className = '' }: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider border rounded ${variantMap[variant]} ${className}`}
    >
      {children}
    </span>
  )
}
