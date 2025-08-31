#!/usr/bin/env python3
"""
Simple TensorBoard Launch Script for Ensemble Stock Trading Project

Usage:
    python3 launch_tensorboard.py [--port PORT] [--host HOST] [--logdir LOGDIR]
    
Example:
    python3 launch_tensorboard.py --port 6006 --host 0.0.0.0
"""

import os
import sys
import argparse
import subprocess
import socket
from pathlib import Path

def check_port_available(host, port):
    """Check if a port is available for binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except socket.error:
            return False

def find_available_port(host, start_port=6006, end_port=6020):
    """Find an available port in the given range."""
    for port in range(start_port, end_port + 1):
        if check_port_available(host, port):
            return port
    return None

def launch_tensorboard(logdir, host='0.0.0.0', port=6006, auto_port=True):
    """Launch TensorBoard with proper configuration."""
    
    # Verify logdir exists
    if not os.path.exists(logdir):
        print(f"Error: Log directory '{logdir}' does not exist!")
        print("Available directories:")
        parent_dir = os.path.dirname(logdir)
        if os.path.exists(parent_dir):
            for item in os.listdir(parent_dir):
                item_path = os.path.join(parent_dir, item)
                if os.path.isdir(item_path):
                    print(f"  - {item_path}")
        return False
    
    # Find available port if auto_port is enabled
    if auto_port and not check_port_available(host, port):
        print(f"Port {port} is not available, searching for alternative...")
        alt_port = find_available_port(host, port, port + 20)
        if alt_port:
            port = alt_port
            print(f"Using port {port} instead")
        else:
            print(f"No available ports found in range {port}-{port+20}")
            return False
    
    # Construct TensorBoard command
    cmd = [
        sys.executable, '-m', 'tensorboard.main',
        '--logdir', logdir,
        '--host', host,
        '--port', str(port),
        '--reload_interval', '10'
    ]
    
    print("=" * 60)
    print("LAUNCHING TENSORBOARD")
    print("=" * 60)
    print(f"Log directory: {logdir}")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"URL: http://{host}:{port}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    print()
    print("Enhanced Metrics Available:")
    print("  📊 Training: reward stats, learning rate, losses")
    print("  📈 Portfolio: Sharpe ratio, max drawdown, returns")  
    print("  🎯 Actions: mean, std, sparsity analysis")
    print("  📺 Episodes: length, reward, count tracking")
    print("  🔧 Diagnostics: gradient norms, reward std")
    print("=" * 60)
    print()
    print("Press Ctrl+C to stop TensorBoard")
    print()
    
    try:
        # Launch TensorBoard
        subprocess.run(cmd)
        return True
    except KeyboardInterrupt:
        print("\nTensorBoard stopped by user")
        return True
    except Exception as e:
        print(f"Error launching TensorBoard: {e}")
        return False

def main():
    """Main function to parse arguments and launch TensorBoard."""
    parser = argparse.ArgumentParser(
        description="Launch TensorBoard for Ensemble Stock Trading Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 launch_tensorboard.py
  python3 launch_tensorboard.py --port 6007
  python3 launch_tensorboard.py --host localhost --port 6006
  python3 launch_tensorboard.py --logdir tensorboard_log/legacy
        """
    )
    
    # Get default logdir
    script_dir = Path(__file__).parent.absolute()
    default_logdir = script_dir / "tensorboard_log"
    
    parser.add_argument(
        '--logdir', 
        type=str, 
        default=str(default_logdir),
        help=f'TensorBoard log directory (default: {default_logdir})'
    )
    parser.add_argument(
        '--host', 
        type=str, 
        default='0.0.0.0',
        help='Host to bind TensorBoard server (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=6006,
        help='Port to bind TensorBoard server (default: 6006)'
    )
    parser.add_argument(
        '--no-auto-port', 
        action='store_true',
        help='Disable automatic port finding if specified port is unavailable'
    )
    
    args = parser.parse_args()
    
    # Launch TensorBoard
    success = launch_tensorboard(
        logdir=args.logdir,
        host=args.host,
        port=args.port,
        auto_port=not args.no_auto_port
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()