// Typed reader for the runtime UI-mode flag. The value is injected before the
// bundle loads (see index.html + docker-entrypoint.sh / vite.config.ts dev
// middleware) as `window.__KIOSK_UI_MODE__`. Any missing/unrecognised value
// falls back to 'operator' so a misconfigured container never accidentally
// serves the customer kiosk screen in place of the trusted operator UI.
export type UiMode = 'operator' | 'customer';

declare global {
  interface Window {
    __KIOSK_UI_MODE__?: string;
  }
}

export function getUiMode(): UiMode {
  return window.__KIOSK_UI_MODE__ === 'customer' ? 'customer' : 'operator';
}
