import { useMemo, useRef, useCallback, useState, type ReactNode } from 'react'

export interface Column<T> {
  key: string
  label: string
  width?: number
  render?: (value: unknown, row: T) => ReactNode
  align?: 'left' | 'right' | 'center'
}

interface VirtualTableProps<T> {
  data: T[]
  columns: Column<T>[]
  rowHeight?: number
  visibleRows?: number
  getRowKey: (row: T) => string | number
  className?: string
}

export default function VirtualTable<T = any>({
  data,
  columns,
  rowHeight = 28,
  visibleRows = 15,
  getRowKey,
  className = '',
}: VirtualTableProps<T>) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [scrollTop, setScrollTop] = useState(0)

  const totalHeight = data.length * rowHeight
  const containerHeight = Math.min(visibleRows * rowHeight, totalHeight)
  const startIndex = Math.floor(scrollTop / rowHeight)
  const endIndex = Math.min(startIndex + visibleRows + 2, data.length)

  const visibleData = useMemo(
    () => data.slice(startIndex, endIndex),
    [data, startIndex, endIndex]
  )

  const onScroll = useCallback(() => {
    if (containerRef.current) {
      setScrollTop(containerRef.current.scrollTop)
    }
  }, [])

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-20 text-gray-500 text-xs font-data uppercase">
        No data
      </div>
    )
  }

  return (
    <div className={`overflow-hidden ${className}`}>
      {/* Header */}
      <div className="flex border-b border-border bg-black/20 text-[10px] text-gray-400 uppercase font-data">
        {columns.map((col) => (
          <div
            key={col.key}
            className={`px-2 py-1.5 truncate ${col.align === 'right' ? 'text-right' : col.align === 'center' ? 'text-center' : 'text-left'}`}
            style={{ width: col.width || `${100 / columns.length}%`, minWidth: col.width || 0 }}
          >
            {col.label}
          </div>
        ))}
      </div>

      {/* Virtual body */}
      <div
        ref={containerRef}
        className="overflow-y-auto"
        style={{ height: containerHeight }}
        onScroll={onScroll}
      >
        <div style={{ height: totalHeight, position: 'relative' }}>
          {visibleData.map((row, i) => {
            const rowIndex = startIndex + i
            return (
              <div
                key={getRowKey(row)}
                className="flex absolute w-full border-b border-border/30 hover:bg-primary/5 transition-colors"
                style={{ top: rowIndex * rowHeight, height: rowHeight }}
              >
                {columns.map((col) => (
                  <div
                    key={col.key}
                    className={`px-2 py-1 text-[11px] font-data truncate ${col.align === 'right' ? 'text-right' : col.align === 'center' ? 'text-center' : 'text-left'}`}
                    style={{ width: col.width || `${100 / columns.length}%`, minWidth: col.width || 0 }}
                  >
                    {col.render
                      ? col.render((row as any)[col.key], row)
                      : String((row as any)[col.key] ?? '')}
                  </div>
                ))}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
