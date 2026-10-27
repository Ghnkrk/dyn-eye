// Owns the single WebSocket connection to the backend (/ws/stream) and
// fans events out:
//   - "logs" frames accumulate into a capped ring buffer for the live log
//     panel (components/dashboard/log-panel.tsx)
//   - "pipeline-status" frames replace the status snapshot directly
//   - "clusters"/"models" frames just invalidate the matching TanStack
//     Query cache entries, so components re-fetch via the normal query
//     hooks instead of keeping a second copy of that data in the store
import { create } from "zustand";

import { queryClient } from "@/lib/queryClient";
import type { LogEvent, PipelineStatus, WsFrame } from "@/lib/types";

const WS_URL =
  process.env.NEXT_PUBLIC_WS_URL ?? (typeof window !== "undefined" ? `ws://${window.location.hostname}:8501/ws/stream` : "");
const MAX_LOGS = 500;
const RECONNECT_DELAY_MS = 2000;

interface LiveState {
  connected: boolean;
  logs: LogEvent[];
  pipelineStatus: PipelineStatus | null;
  connect: () => void;
  clearLogs: () => void;
}

let socket: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

export const useLiveStore = create<LiveState>((set, get) => ({
  connected: false,
  logs: [],
  pipelineStatus: null,

  clearLogs: () => set({ logs: [] }),

  connect: () => {
    if (socket || !WS_URL) return;

    const ws = new WebSocket(WS_URL);
    socket = ws;

    ws.onopen = () => {
      set({ connected: true });
      ws.send(JSON.stringify({ subscribe: ["logs", "pipeline-status", "clusters", "models"] }));
    };

    ws.onmessage = (event) => {
      let frame: WsFrame;
      try {
        frame = JSON.parse(event.data);
      } catch {
        return;
      }

      switch (frame.topic) {
        case "logs": {
          const next = [...get().logs, frame.data as LogEvent].slice(-MAX_LOGS);
          set({ logs: next });
          break;
        }
        case "pipeline-status": {
          set({ pipelineStatus: frame.data as PipelineStatus });
          queryClient.setQueryData(["pipeline", "status"], frame.data);
          break;
        }
        case "clusters": {
          queryClient.invalidateQueries({ queryKey: ["clusters"] });
          break;
        }
        case "models": {
          queryClient.invalidateQueries({ queryKey: ["models"] });
          break;
        }
      }
    };

    ws.onclose = () => {
      socket = null;
      set({ connected: false });
      reconnectTimer = setTimeout(() => get().connect(), RECONNECT_DELAY_MS);
    };

    ws.onerror = () => {
      ws.close();
    };
  },
}));

export function disconnectLiveStore(): void {
  if (reconnectTimer) clearTimeout(reconnectTimer);
  socket?.close();
  socket = null;
}
