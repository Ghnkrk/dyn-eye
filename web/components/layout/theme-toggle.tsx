"use client";

import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { useSyncExternalStore } from "react";

import { Button } from "@/components/ui/button";

const emptySubscribe = () => () => {};

export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  // Avoid rendering theme-dependent UI before hydration (resolvedTheme is
  // undefined on first render) to prevent a server/client mismatch — true
  // during SSR/first render, false once mounted on the client.
  const isServer = useSyncExternalStore(
    emptySubscribe,
    () => false,
    () => true
  );

  if (isServer) {
    return <Button variant="ghost" size="icon" aria-hidden className="pointer-events-none opacity-0" />;
  }

  const isDark = resolvedTheme === "dark";

  return (
    <Button
      variant="ghost"
      size="icon"
      aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
      onClick={() => setTheme(isDark ? "light" : "dark")}
    >
      {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </Button>
  );
}
