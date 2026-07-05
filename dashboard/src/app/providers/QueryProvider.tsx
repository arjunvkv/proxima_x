import { type ReactNode } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 10_000,
      retry: 2,
      refetchOnWindowFocus: false,
    },
  },
})

interface Props {
  children: ReactNode
}

export default function QueryProvider({ children }: Props) {
  return (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  )
}
