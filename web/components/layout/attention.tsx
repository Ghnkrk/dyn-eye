"use client";

import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Bell, CircleAlert } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { attentionApi } from "@/lib/api";
import { queryClient } from "@/lib/queryClient";
import type { AttentionItem } from "@/lib/types";
import { cn } from "@/lib/utils";

const TOASTED_KEY = "dyneye-attention-toasted";

export function useAttention() {
  const { data } = useQuery({
    queryKey: ["attention"],
    queryFn: attentionApi.list,
    refetchInterval: 30000, // backstop; WebSocket frames refresh it immediately
  });
  return { items: data?.items ?? [], blocking: data?.blocking ?? 0 };
}

/** True when an item points at this page (exact for "/", prefix otherwise). */
function onPage(item: AttentionItem, pathname: string) {
  return item.page === "/" ? pathname === "/" : pathname.startsWith(item.page);
}

export function usePageAttention(pathname: string) {
  const { items } = useAttention();
  return items.filter((i) => onPage(i, pathname));
}

/** Sidebar dot colour for a nav entry, or null when nothing waits there. */
export function useNavAttention(): (href: string) => "blocking" | "recommended" | null {
  const { items } = useAttention();
  return (href) => {
    const here = items.filter((i) => (href === "/" ? i.page === "/" : i.page.startsWith(href)));
    if (here.some((i) => i.severity === "blocking")) return "blocking";
    return here.length ? "recommended" : null;
  };
}

async function dismiss(item: AttentionItem, snoozeMinutes?: number) {
  try {
    await attentionApi.dismiss(item.key, snoozeMinutes);
    queryClient.invalidateQueries({ queryKey: ["attention"] });
  } catch (e) {
    toast.error(e instanceof Error ? e.message : "Could not update the reminder");
  }
}

function since(iso: string | null) {
  if (!iso) return "";
  const mins = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min ago`;
  const hours = Math.round(mins / 60);
  if (hours < 48) return `${hours} h ago`;
  return `${Math.round(hours / 24)} days ago`;
}

function SeverityIcon({ item, className }: { item: AttentionItem; className?: string }) {
  return item.severity === "blocking" ? (
    <CircleAlert className={cn("h-4 w-4 shrink-0 text-destructive", className)} />
  ) : (
    <AlertTriangle className={cn("h-4 w-4 shrink-0 text-amber-500", className)} />
  );
}

function ItemActions({ item, onGo }: { item: AttentionItem; onGo?: () => void }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button size="sm" nativeButton={false} render={<Link href={item.href} onClick={onGo} />}>
        {item.action}
      </Button>
      <Button size="sm" variant="ghost" onClick={() => dismiss(item, 60)}>
        Snooze 1 h
      </Button>
      <Button size="sm" variant="ghost" onClick={() => dismiss(item)}>
        Dismiss
      </Button>
    </div>
  );
}

/** Bell + slide-over list in the top bar. */
export function AttentionBell() {
  const { items, blocking } = useAttention();
  const count = items.length;

  return (
    <Sheet>
      <SheetTrigger
        render={
          <Button
            variant="ghost"
            size="icon"
            className="relative"
            aria-label={count ? `${count} item${count === 1 ? " needs" : "s need"} your input` : "No items need your input"}
          />
        }
      >
        <Bell className={cn("h-4 w-4", blocking > 0 && "text-destructive")} />
        {count > 0 && (
          <span
            className={cn(
              "absolute -right-0.5 -top-0.5 flex h-4 min-w-4 items-center justify-center rounded-full px-1 text-[10px] font-semibold text-white",
              blocking > 0 ? "bg-destructive" : "bg-amber-500"
            )}
          >
            {count}
          </span>
        )}
      </SheetTrigger>
      <SheetContent className="flex w-full flex-col gap-0 sm:max-w-md">
        <SheetHeader>
          <SheetTitle>Needs your input</SheetTitle>
          <SheetDescription>
            {count === 0
              ? "Nothing is waiting on you. The system will tell you here when it needs a decision."
              : blocking > 0
                ? `${blocking} item${blocking === 1 ? "" : "s"} block${blocking === 1 ? "s" : ""} the system from moving on.`
                : "Recommended next steps."}
          </SheetDescription>
        </SheetHeader>
        <ul className="flex flex-col gap-3 overflow-y-auto px-4 pb-4">
          {items.map((item) => (
            <li
              key={item.key}
              className={cn(
                "flex flex-col gap-2 rounded-lg border p-3",
                item.severity === "blocking" && "border-destructive/50 bg-destructive/5"
              )}
            >
              <div className="flex items-start gap-2">
                <SeverityIcon item={item} className="mt-0.5" />
                <div className="flex min-w-0 flex-col gap-0.5">
                  <span className="text-sm font-medium">{item.title}</span>
                  <span className="text-xs text-muted-foreground">{item.detail}</span>
                  <span className="text-[11px] text-muted-foreground">
                    {item.severity === "blocking" ? "Blocking" : "Recommended"}
                    {item.since && ` · ${since(item.since)}`}
                  </span>
                </div>
              </div>
              <ItemActions item={item} />
            </li>
          ))}
        </ul>
      </SheetContent>
    </Sheet>
  );
}

/** Toasts each new item once (remembered per browser). */
export function AttentionToasts() {
  const { items } = useAttention();
  const router = useRouter();

  useEffect(() => {
    let toasted: string[] = [];
    try {
      toasted = JSON.parse(localStorage.getItem(TOASTED_KEY) ?? "[]");
    } catch {
      toasted = [];
    }
    const fresh = items.filter((i) => !toasted.includes(i.key));
    if (!fresh.length) return;
    for (const item of fresh) {
      const show = item.severity === "blocking" ? toast.error : toast.warning;
      show(item.title, {
        description: item.detail,
        duration: item.severity === "blocking" ? 15000 : 8000,
        action: { label: item.action, onClick: () => router.push(item.href) },
      });
    }
    // keep only keys that still exist, so the list never grows without bound
    const live = new Set(items.map((i) => i.key));
    try {
      localStorage.setItem(
        TOASTED_KEY,
        JSON.stringify([...toasted.filter((k) => live.has(k)), ...fresh.map((i) => i.key)])
      );
    } catch {
      // storage unavailable: worst case the toast shows again next visit
    }
  }, [items, router]);

  return null;
}

/** Banner for the items that point at the current page. */
export function AttentionBanner({ pathname }: { pathname: string }) {
  const items = usePageAttention(pathname);
  if (!items.length) return null;
  return (
    <div className="flex flex-col gap-2">
      {items.map((item) => (
        <div
          key={item.key}
          role="status"
          className={cn(
            "flex flex-col gap-2 rounded-lg border px-4 py-3 sm:flex-row sm:items-center sm:justify-between",
            item.severity === "blocking"
              ? "border-destructive/50 bg-destructive/5"
              : "border-amber-500/40 bg-amber-500/5"
          )}
        >
          <div className="flex items-start gap-2">
            <SeverityIcon item={item} className="mt-0.5" />
            <div className="flex flex-col gap-0.5">
              <span className="text-sm font-medium">{item.title}</span>
              <span className="text-xs text-muted-foreground">{item.detail}</span>
            </div>
          </div>
          {item.href !== pathname ? (
            <ItemActions item={item} />
          ) : (
            <div className="flex gap-2">
              <Button size="sm" variant="ghost" onClick={() => dismiss(item, 60)}>
                Snooze 1 h
              </Button>
              <Button size="sm" variant="ghost" onClick={() => dismiss(item)}>
                Dismiss
              </Button>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

/** Banner slot for the app layout: shows what the current page is waiting for. */
export function PageAttentionBanner() {
  const pathname = usePathname();
  return (
    <div className="mb-6 empty:mb-0">
      <AttentionBanner pathname={pathname} />
    </div>
  );
}

/** Mark an item done when the user does what it asked (e.g. opens the image viewer). */
export function acknowledge(keyPrefix: string, items: AttentionItem[]) {
  const item = items.find((i) => i.key.startsWith(keyPrefix));
  if (item) void dismiss(item);
}
