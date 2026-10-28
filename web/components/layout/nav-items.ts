import { Activity, Boxes, Flame, GitCompareArrows, LayoutDashboard, ListTree } from "lucide-react";

export const NAV_ITEMS = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/clusters", label: "Clusters", icon: Boxes },
  { href: "/finetuning", label: "FineTuning", icon: Flame },
  { href: "/models", label: "Models", icon: GitCompareArrows },
  { href: "/inference", label: "Inference", icon: Activity },
  { href: "/runs", label: "Runs", icon: ListTree },
] as const;
