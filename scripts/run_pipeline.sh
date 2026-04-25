#!/bin/bash

# L.E.O. Data Pipeline
# Orchestrates the entire data generation flow: PDF -> Web -> Gen -> Format

set -e # Stop on error

# Define project root
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_ROOT"

echo "🦁 L.E.O. Data Factory Pipeline"
echo "================================="
echo "Project Root: $PROJECT_ROOT"
echo ""

# Run Full Pipeline
echo "🚀 Starting Integrated Data Pipeline..."
python scripts/leo.py data full

echo ""
echo "🎉 PIPELINE FINISHED SUCCESSFULLY!"
echo "👉 Data is available in data/synthetic/"
