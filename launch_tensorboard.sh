#!/bin/bash
# Alternative bash script to launch TensorBoard with proper environment setup
# This script works around Python environment issues that may prevent TensorBoard from starting

set -e

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up TensorBoard processes..."
    CLEANUP_PIDS=$(pgrep -f tensorboard 2>/dev/null || true)
    if [ -n "$CLEANUP_PIDS" ]; then
        echo "Killing TensorBoard processes: $CLEANUP_PIDS"
        echo "$CLEANUP_PIDS" | xargs kill 2>/dev/null || true
        sleep 1
        echo "✓ TensorBoard processes terminated"
    else
        echo "No TensorBoard processes to clean up"
    fi
}

# Setup signal handlers
trap cleanup EXIT
trap cleanup SIGINT
trap cleanup SIGTERM

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="${SCRIPT_DIR}/tensorboard_log"

# Default values
HOST="0.0.0.0"
PORT="6006"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --logdir)
            LOGDIR="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--logdir DIR] [--host HOST] [--port PORT]"
            echo ""
            echo "Options:"
            echo "  --logdir DIR    TensorBoard log directory (default: ./tensorboard_log)"
            echo "  --host HOST     Host to bind TensorBoard server (default: 0.0.0.0)"
            echo "  --port PORT     Port to bind TensorBoard server (default: 6006)"
            echo "  --help,-h       Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --port 6007"
            echo "  $0 --logdir tensorboard_log/legacy --port 6008"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Kill any existing TensorBoard processes
echo "Checking for existing TensorBoard processes..."
EXISTING_PIDS=$(pgrep -f tensorboard 2>/dev/null || true)
if [ -n "$EXISTING_PIDS" ]; then
    echo "Found existing TensorBoard process(es): $EXISTING_PIDS"
    echo "Killing existing TensorBoard processes..."
    echo "$EXISTING_PIDS" | xargs kill 2>/dev/null || true
    sleep 2
    echo "✓ Existing TensorBoard processes terminated"
    echo ""
else
    echo "No existing TensorBoard processes found"
    echo ""
fi

# Check if logdir exists
if [ ! -d "$LOGDIR" ]; then
    echo "Error: Log directory '$LOGDIR' does not exist!"
    echo ""
    echo "Available directories in $(dirname "$LOGDIR"):"
    if [ -d "$(dirname "$LOGDIR")" ]; then
        ls -la "$(dirname "$LOGDIR")" | grep "^d"
    fi
    exit 1
fi

# Find available Python/TensorBoard command
TENSORBOARD_CMD=""

# Try different TensorBoard commands
if command -v tensorboard &> /dev/null; then
    TENSORBOARD_CMD="tensorboard"
elif python3 -c "import tensorboard" 2>/dev/null; then
    TENSORBOARD_CMD="python3 -m tensorboard.main"
elif python -c "import tensorboard" 2>/dev/null; then
    TENSORBOARD_CMD="python -m tensorboard.main"
else
    echo "Error: TensorBoard not found!"
    echo "Please install TensorBoard: pip install tensorboard"
    exit 1
fi

echo "========================================="
echo "LAUNCHING TENSORBOARD"
echo "========================================="
echo "Log directory: $LOGDIR"
echo "Host: $HOST"
echo "Port: $PORT"
echo "URL: http://$HOST:$PORT"
echo "Command: $TENSORBOARD_CMD"
echo "========================================="
echo ""
echo "Enhanced Metrics Available:"
echo "📊 Training: reward stats, learning rate, losses"
echo "📈 Portfolio: Sharpe ratio, max drawdown, returns"  
echo "🎯 Actions: mean, std, sparsity analysis"
echo "📺 Episodes: length, reward, count tracking"
echo "🔧 Diagnostics: gradient norms, reward std"
echo "========================================="
echo ""
echo "Press Ctrl+C to stop TensorBoard"
echo ""

# Set environment variables
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Launch TensorBoard
exec $TENSORBOARD_CMD \
    --logdir="$LOGDIR" \
    --host="$HOST" \
    --port="$PORT" \
    --reload_interval=10 \
    --max_reload_threads=4