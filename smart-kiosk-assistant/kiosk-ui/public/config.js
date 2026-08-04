// Dev-server default for the runtime UI-mode flag (see docker-entrypoint.sh).
// In production this file is regenerated at container start from
// $KIOSK_UI_MODE; in `vite dev` it is served as-is from /public, so change
// the value below (or run `KIOSK_UI_MODE=customer npm run dev` — see
// vite.config.ts) to preview the customer kiosk screen locally.
window.__KIOSK_UI_MODE__ = 'operator';
