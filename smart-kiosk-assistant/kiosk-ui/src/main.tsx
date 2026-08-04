import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { AuthGate } from './components/Auth/AuthGate';
import CustomerApp from './components/Customer/CustomerApp';
import { getUiMode } from './uiMode';
import './index.css';

// Single build, single image — which screen renders is decided at container
// start by KIOSK_UI_MODE (see uiMode.ts, docker-entrypoint.sh). 'operator'
// (default) is the existing chat + performance dashboard behind AuthGate,
// unchanged. 'customer' renders the new kiosk-facing ordering screen.
const root = createRoot(document.getElementById('root')!);
if (getUiMode() === 'customer') {
  root.render(
    <StrictMode>
      <CustomerApp />
    </StrictMode>,
  );
} else {
  root.render(
    <StrictMode>
      <AuthGate>
        <App />
      </AuthGate>
    </StrictMode>,
  );
}
