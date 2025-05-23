#!/usr/bin/env python3
"""
Quick launcher for the RL Swarm Demo
Run this to start the interactive demonstration
"""

import sys
import subprocess
import importlib.util

def check_dependencies():
    """Check if required packages are installed"""
    required = ['pygame', 'numpy']
    missing = []
    
    for package in required:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing.append(package)
    
    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print("Please install with: pip install pygame numpy")
        return False
    return True

def main():
    print("🤖 Starting RL Swarm Learning Demo...")
    print("="*50)
    
    if not check_dependencies():
        return
    
    try:
        from rl_game import RLSwarmGame
        
        print("Controls:")
        print("- Start/Pause: Control simulation")  
        print("- Restart: Reset to generation 0")
        print("- Edit Obstacles: Add/remove obstacles")
        print("- Watch the swarm learn over time!")
        print("="*50)
        
        game = RLSwarmGame()
        game.run()
        
    except ImportError as e:
        print(f"Error importing game: {e}")
        print("Make sure rl_game.py is in the same directory")
    except Exception as e:
        print(f"Error running demo: {e}")

if __name__ == "__main__":
    main() 