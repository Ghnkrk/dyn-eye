"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { useNavAttention } from "@/components/layout/attention";
import { NAV_ITEMS } from "@/components/layout/nav-items";
import { cn } from "@/lib/utils";

export function Sidebar() {
  const pathname = usePathname();
  const attention = useNavAttention();

  return (
    <aside className="hidden w-56 flex-col border-r bg-sidebar p-4 md:flex">
      <div className="mb-6 px-2">
        <p className="text-sm font-semibold tracking-tight">DYN-EYE</p>
        <p className="text-xs text-muted-foreground">Defect discovery</p>
      </div>
      <nav className="flex flex-col gap-1">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          const waiting = attention(href);
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-2 rounded-md px-2 py-2 text-sm font-medium transition-colors",
                active ? "bg-primary/10 text-primary" : "text-muted-foreground hover:bg-accent hover:text-foreground"
              )}
            >
              <Icon className="h-4 w-4" />
              {label}
              {waiting && (
                <span
                  title={waiting === "blocking" ? "Needs your input" : "Recommended action waiting"}
                  className={cn(
                    "ml-auto h-2 w-2 rounded-full",
                    waiting === "blocking" ? "bg-destructive" : "bg-amber-500"
                  )}
                />
              )}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
