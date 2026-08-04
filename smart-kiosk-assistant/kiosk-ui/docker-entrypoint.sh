#!/bin/sh
# Writes a tiny runtime config file the SPA reads before the React bundle
# loads, so a single built image can serve either the operator UI (chat +
# performance dashboard) or the customer-facing kiosk UI, selected purely by
# the KIOSK_UI_MODE environment variable on the container -- no rebuild.
#
# Installed as /docker-entrypoint.d/40-kiosk-ui-mode.sh: the official nginx
# image's own entrypoint (still the container's actual ENTRYPOINT) runs every
# executable script in that directory before exec'ing the CMD, so this file
# must only perform its side effect and return -- it must NOT exec/replace
# the process itself.
set -e

MODE="${KIOSK_UI_MODE:-operator}"

cat > /usr/share/nginx/html/config.js <<EOF
window.__KIOSK_UI_MODE__ = "${MODE}";
EOF
