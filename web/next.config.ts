import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Lean runtime image for Docker (web/Dockerfile) — copies only the
  // standalone server output instead of full node_modules.
  output: "standalone",

  // This app is a fully live, mutation-heavy operational dashboard — every
  // page is client-rendered against the FastAPI backend (TanStack Query +
  // WebSocket), so there's no static content for Cache Components'
  // prerendering model to help with. Using the classic dynamic-by-default
  // fetch model instead avoids Suspense/`use cache` ceremony that doesn't
  // apply to this app. See node_modules/next/dist/docs/01-app/02-guides/
  // caching-without-cache-components.md.
  cacheComponents: false,

  // tw-animate-css's package.json only declares a "style" export condition
  // with no "default"/"require" fallback, which Turbopack's CSS resolver
  // doesn't recognize yet (fails even plain `require.resolve` in Node).
  // Alias the bare specifier straight to its compiled CSS file.
  turbopack: {
    resolveAlias: {
      "tw-animate-css": "./node_modules/tw-animate-css/dist/tw-animate.css",
    },
    // Required for Tailwind v4 under Turbopack — create-next-app's scaffold
    // wires this up automatically; don't drop it on a config rewrite.
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },

  // REST calls go through this same-origin proxy (avoids CORS entirely).
  // The WebSocket connects directly to the backend (see lib/ws.ts) —
  // Next's rewrite proxy's WebSocket-upgrade support varies by deploy
  // target, so it isn't worth relying on for a dashboard with no
  // user-facing cross-origin exposure anyway (CORS is wide open backend-side).
  async rewrites() {
    const apiBase = process.env.API_PROXY_TARGET ?? "http://localhost:8501";
    return [{ source: "/api/v2/:path*", destination: `${apiBase}/api/v2/:path*` }];
  },

  experimental: {
    // Next buffers proxied request bodies (to allow re-reading them) up to
    // this cap, defaulting to 10MB — way too small for the discovery folder
    // upload (/api/v2/images/upload), which can be a full batch of
    // industrial inspection images. Raised generously since this is an
    // internal tool, not a public endpoint.
    proxyClientMaxBodySize: "2gb",
  },
};

export default nextConfig;
