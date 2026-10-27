"use client";

import { QueryClientProvider } from "@tanstack/react-query";
import { useEffect } from "react";

import { queryClient } from "@/lib/queryClient";
import { useLiveStore } from "@/store/useLiveStore";

export function Providers({ children }: { children: React.ReactNode }) {
  const connect = useLiveStore((s) => s.connect);

  useEffect(() => {
    connect();
  }, [connect]);

  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}
