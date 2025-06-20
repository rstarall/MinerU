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

# Check if models are available (for local model source)
if [ "${MINERU_MODEL_SOURCE:-}" = "local" ]; then
    echo "Checking local models..."
    python3 -c "
try:
    from mineru.utils.config_reader import get_local_models_dir
    models_config = get_local_models_dir()
    if models_config:
        print('✓ Local models configuration found')
        if 'vlm' in models_config:
            print(f'✓ VLM models path: {models_config[\"vlm\"]}')
        if 'pipeline' in models_config:
            print(f'✓ Pipeline models path: {models_config[\"pipeline\"]}')
    else:
        print('⚠ No local models configuration found')
except Exception as e:
    print(f'⚠ Error checking models: {e}')
"
fi

echo "==========================================="
echo "Starting FastAPI server..."
echo "Access the API at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "==========================================="

# Start the FastAPI application
exec uvicorn app:app "$@" 