#!/bin/bash
# Post-start script for development container

set -e

echo "🔄 Running post-start setup..."

# Check if all services are ready
echo "⚡ Checking service health..."

# Wait for any background services to start
sleep 2

# Verify Python environment
if python -c "import torch; print('PyTorch:', torch.__version__)" > /dev/null 2>&1; then
    echo "✅ PyTorch is ready"
else
    echo "❌ PyTorch not available"
fi

# Check if package is installed correctly
if python -c "import pno_physics_bench; print('PNO Physics Bench version:', pno_physics_bench.__version__)" > /dev/null 2>&1; then
    echo "✅ PNO Physics Bench package is ready"
else
    echo "⚠️  PNO Physics Bench package not yet implemented (expected for new repository)"
fi

# Start background services if needed
echo "🚀 Starting background services..."

# Start TensorBoard in background (if logs exist)
if [ -d "results/tensorboard_logs" ] && [ "$(ls -A results/tensorboard_logs)" ]; then
    echo "📊 Starting TensorBoard..."
    nohup tensorboard --logdir=results/tensorboard_logs --host=0.0.0.0 --port=6006 > logs/tensorboard.log 2>&1 &
    echo "   TensorBoard available at http://localhost:6006"
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Display useful information
echo ""
echo "🎯 Development Environment Ready!"
echo "=================================="
echo "📁 Workspace: $(pwd)"
echo "🐍 Python: $(python --version)"
echo "📦 Pip packages: $(pip list | wc -l) installed"
echo ""
echo "🚀 Quick Commands:"
echo "   make test          - Run tests"
echo "   make format        - Format code"
echo "   make docs          - Build documentation"
echo "   jupyter lab        - Start Jupyter Lab"
echo ""
echo "🌐 Available Ports:"
echo "   8888 - Jupyter Lab"
echo "   6006 - TensorBoard"
echo "   8000 - Documentation"
echo "   5000 - API Server"
echo ""

# Check for updates to requirements
if [ -f "requirements.txt" ]; then
    echo "🔍 Checking for dependency updates..."
    pip list --outdated --format=columns | head -10
fi

# Display git status if in a git repository
if [ -d ".git" ]; then
    echo "📝 Git Status:"
    git status --porcelain | head -5
    if [ $(git status --porcelain | wc -l) -gt 5 ]; then
        echo "   ... and $(( $(git status --porcelain | wc -l) - 5 )) more files"
    fi
    echo ""
fi

# Check disk space
echo "💾 Disk Usage:"
df -h /app | tail -1

echo ""
echo "✨ Ready for development!"