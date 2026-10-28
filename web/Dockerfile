# Multi-stage build producing a lean Next.js standalone runtime image.

FROM node:22-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:22-slim AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# next.config.ts's rewrites() is evaluated at build time (next build bakes
# its return value into .next/routes-manifest.json) — not per-request — so
# API_PROXY_TARGET must be available here, not just at container runtime.
ARG API_PROXY_TARGET=http://api:8501
ENV API_PROXY_TARGET=${API_PROXY_TARGET}
RUN npm run build

FROM node:22-slim AS runner
WORKDIR /app
ENV NODE_ENV=production

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000
CMD ["node", "server.js"]
