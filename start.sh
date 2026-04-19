#!/usr/bin/env bash
set -euo pipefail

echo "[start] launching FastAPI on :8000"
uvicorn api.main:app --host 127.0.0.1 --port 8000 --log-level info &
API_PID=$!

echo "[start] waiting for API to be ready"
for i in {1..60}; do
    if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
        echo "[start] API is ready"
        break
    fi
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo "[start] API process died before becoming ready"
        exit 1
    fi
    sleep 2
done

echo "[start] launching Streamlit on :${STREAMLIT_SERVER_PORT:-7860}"
exec streamlit run ui/app.py
