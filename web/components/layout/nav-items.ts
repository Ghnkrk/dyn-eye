import { Activity, Boxes, GitCompareArrows, LayoutDashboard, ListTree } from "lucide-react";

export const NAV_ITEMS = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/clusters", label: "Clusters", icon: Boxes },
  { href: "/models", label: "Models", icon: GitCompareArrows },
  { href: "/inference", label: "Inference", icon: Activity },
  { href: "/runs", label: "Runs", icon: ListTree },
] as const;
