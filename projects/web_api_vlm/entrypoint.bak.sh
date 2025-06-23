#!/usr/bin/env bash
set -euo pipefail

# Print startup information
echo "==========================================="
echo "MinerU VLM Web API Starting..."
echo "==========================================="
echo "Environment: $(python3 --version)"
echo "MinerU Version: $(python3 -c 'import mineru; print(mineru.__version__)' 2>/dev/null || echo 'Unknown')"
echo "Working Directory: $(pwd)"
echo "Model Source: ${MINERU_MODEL_SOURCE:-auto}"
echo "==========================================="

# Validate model configuration
echo "Validating model configuration..."
if [ -f "/root/mineru.json" ]; then
    echo "✓ Configuration file found at /root/mineru.json"
else
    echo "⚠ Configuration file not found, using default settings"
fi

# Run model download and check script
echo "Running model download and check script..."
python3 /app/model_download.py

echo "==========================================="
echo "Starting FastAPI server..."
echo "Access the API at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "==========================================="

# Start the FastAPI application
exec uvicorn app:app "$@" 