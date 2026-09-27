#!/bin/sh
set -e

# Run DB migrations on startup
alembic -c /app/alembic.ini upgrade head

# Start API server in background
compact-rag serve --host 0.0.0.0 --port 8000 &

# Start Streamlit admin dashboard in foreground (keeps container alive)
exec compact-rag admin --host 0.0.0.0 --port 8501
