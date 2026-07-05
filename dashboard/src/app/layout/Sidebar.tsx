import { NavLink } from 'react-router-dom'

interface NavItem {
  path: string
  label: string
  icon: string
}

const navItems: NavItem[] = [
  { path: '/', label: 'Dashboard', icon: 'grid_view' },
  { path: '/assets', label: 'Assets', icon: 'currency_exchange' },
  { path: '/intelligence', label: 'Intelligence', icon: 'cyclone' },
  { path: '/entropy', label: 'Entropy', icon: 'blur_on' },
  { path: '/thermo', label: 'Thermo', icon: 'device_thermostat' },
  { path: '/tpi', label: 'TPI Flow', icon: 'swap_calls' },
  { path: '/research', label: 'Research', icon: 'science' },
  { path: '/risk', label: 'Risk', icon: 'shield' },
  { path: '/shader', label: 'Shader', icon: 'visibility' },
]

export default function Sidebar() {
  return (
    <aside className="w-16 lg:w-56 border-r border-border bg-background flex flex-col flex-shrink-0 overflow-y-auto">
      <nav className="flex flex-col py-2">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-4 py-3 text-xs uppercase tracking-widest font-body transition-colors ${
                isActive
                  ? 'text-primary bg-primary/10 border-l-2 border-primary'
                  : 'text-gray-500 hover:text-primary hover:bg-primary/5 border-l-2 border-transparent'
              }`
            }
          >
            <span className="material-symbols-outlined text-lg">{item.icon}</span>
            <span className="hidden lg:inline">{item.label}</span>
          </NavLink>
        ))}
      </nav>
    </aside>
  )
}
