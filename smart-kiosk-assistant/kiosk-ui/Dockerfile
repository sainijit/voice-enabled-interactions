# ── Stage 1: build the Vite/React bundle ──────────────────────────────────────
FROM node:20-alpine AS build
WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install

COPY . .
RUN npm run build

# ── Stage 2: serve the static bundle + reverse-proxy backends with nginx ──────
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
COPY docker-entrypoint.sh /docker-entrypoint.d/40-kiosk-ui-mode.sh
RUN chmod +x /docker-entrypoint.d/40-kiosk-ui-mode.sh

EXPOSE 7860
CMD ["nginx", "-g", "daemon off;"]
