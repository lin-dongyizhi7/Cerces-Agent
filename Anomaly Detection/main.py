#!/usr/bin/env python3
"""
Anomaly detection layer main program
"""

import signal
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from AnomalyDetectionController import AnomalyDetectionController
from ConfigManager import ConfigManager


def signal_handler(sig, frame):
    """Signal handler for graceful shutdown"""
    print("\n收到停止信号，正在关闭...")
    sys.exit(0)


def main():
    """Main function"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Load configuration
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/detection.yaml"
    
    try:
        config = ConfigManager.load(config_path)
    except FileNotFoundError:
        print(f"Warning: Configuration file not found: {config_path}")
        print("Using default configuration...")
        config = {
            'detection_layer': {
                'data_receiver': {
                    'message_queue': {
                        'endpoint': 'tcp://localhost:5555',
                        'buffer_size': 10000
                    }
                },
                'detection_engine': {
                    'statistical': {'enabled': True},
                    'comparison': {'enabled': True},
                    'ml': {'enabled': True}
                },
                'rule_orchestrator': {
                    'enabled_rules': ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']
                },
                'alert': {
                    'dedup_window': 300,
                    'levels': {
                        'critical': ['R1', 'R4'],
                        'warning': ['R2', 'R3'],
                        'info': ['R7', 'R10']
                    }
                }
            }
        }
    
    # Create controller
    controller = AnomalyDetectionController(config)
    
    try:
        # Start anomaly detection layer
        print("启动异常检测层...")
        controller.start()
        print("异常检测层运行中，按Ctrl+C停止...")
        
        # Keep running
        import time
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n正在停止...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        controller.stop()
        print("异常检测层已停止")


if __name__ == "__main__":
    main()

