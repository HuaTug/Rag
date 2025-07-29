#!/bin/bash

# Export Environment Variables Script
# This script exports configuration variables from the .env file

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

# Check if .env file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found at $ENV_FILE"
    exit 1
fi

echo "Exporting environment variables from $ENV_FILE..."
# Google Search API Configuration
export GOOGLE_API_KEY="AIzaSyCJffa8kg0c1_Ef7zl18QUMZVvqGwBVtrM"
export GOOGLE_SEARCH_ENGINE_ID="e6676dbfd052c4ecf"
# DeepSeek API Configuration
export DEEPSEEK_API_KEY="sk-qFPEqgpxmS8DJ0nJQ6gvdIkozY1k2oEZER2A4zRhLxBvtIHl"

echo "Environment variables exported successfully!"
