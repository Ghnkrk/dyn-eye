// Single shared QueryClient instance. Used both by <QueryClientProvider>
// in app/providers.tsx and by store/useLiveStore.ts, which invalidates
// queries in response to WebSocket events arriving outside of React.
import { QueryClient } from "@tanstack/react-query";

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 10_000,
      refetchOnWindowFocus: false,
    },
  },
});
