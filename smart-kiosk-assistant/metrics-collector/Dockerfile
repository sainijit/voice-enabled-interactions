# Dockerfile – metrics-collector for the Smart Kiosk Assistant.
#
# Uses intel/retail-benchmark as the base image, which ships:
#   supervisord      – process manager that starts all collectors
#   qmassa           – Intel GPU utilization (writes qmassa1-*-tool-generated.json)
#   Intel NPU tool   – NPU utilization (writes npu_usage.csv)
#   Intel PCM        – CPU package power (writes pcm.csv)
#   sar / free       – CPU and memory (managed by supervisord.conf)
#
# pciutils is added for lspci (used by /platform-info to detect iGPU/NPU names).

FROM intel/retail-benchmark:3.3.2

ARG HTTP_PROXY

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        pciutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app.py   ./app.py
COPY start.sh /start.sh

RUN chmod +x /start.sh

EXPOSE 9000

# start.sh starts supervisord (base-image collectors) then runs app.py.
CMD ["/start.sh"]
