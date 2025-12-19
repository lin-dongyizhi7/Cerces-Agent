"""
Data receiver - receives data from communication layer
"""

from typing import Callable, List
import sys
import os
import json
import threading
import time

sys.path.append(os.path.dirname(__file__))
from common.data_structures import StructuredData

try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    print("Warning: zmq not available, DataReceiver will not work")


class DataReceiver:
    """Data receiver from communication layer"""
    
    def __init__(self, config: dict):
        """
        Initialize data receiver
        
        Args:
            config: Configuration dictionary
        """
        if not ZMQ_AVAILABLE:
            raise RuntimeError("zmq is required for DataReceiver")
        
        self.config = config
        self.message_queue_config = config.get('message_queue', {})
        self.endpoint = self.message_queue_config.get('endpoint', 'tcp://localhost:5555')
        self.buffer_size = self.message_queue_config.get('buffer_size', 10000)
        
        self.context = None
        self.socket = None
        self.callbacks: List[Callable[[StructuredData], None]] = []
        self.running = False
        self.receive_thread = None
    
    def register_callback(self, callback: Callable[[StructuredData], None]):
        """
        Register callback for received data
        
        Args:
            callback: Callback function
        """
        self.callbacks.append(callback)
    
    def start(self):
        """Start data receiver"""
        if self.running:
            return
        
        if not ZMQ_AVAILABLE:
            raise RuntimeError("zmq is required for DataReceiver")
        
        # Initialize ZeroMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(self.endpoint)
        
        self.running = True
        
        # Start receive thread
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()
        
        print(f"DataReceiver started, listening on {self.endpoint}")
    
    def stop(self):
        """Stop data receiver"""
        self.running = False
        
        if self.socket:
            self.socket.close()
        
        if self.context:
            self.context.term()
        
        if self.receive_thread:
            self.receive_thread.join(timeout=1.0)
        
        print("DataReceiver stopped")
    
    def _receive_loop(self):
        """Receive loop running in separate thread"""
        while self.running:
            try:
                # Receive message (non-blocking with timeout)
                try:
                    message = self.socket.recv(zmq.NOBLOCK)
                except zmq.Again:
                    time.sleep(0.01)  # Small sleep to avoid busy waiting
                    continue
                
                # Parse JSON message
                try:
                    data_dict = json.loads(message.decode('utf-8'))
                    structured_data = StructuredData.from_dict(data_dict)
                    
                    # Call all registered callbacks
                    for callback in self.callbacks:
                        try:
                            callback(structured_data)
                        except Exception as e:
                            print(f"Error in callback: {e}")
                
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"Error parsing message: {e}")
                    continue
            
            except Exception as e:
                if self.running:
                    print(f"Error in receive loop: {e}")
                break

