"""Smart Kiosk domain plugins.

Mounted into the rag-service container at runtime:
  ./plugins:/app/rag-service/plugins:ro

When rag-service moves to edge-ai-libraries, this directory stays in the
smart-kiosk-assistant repo and is mounted the same way — zero changes to the
rag-service image needed.
"""
