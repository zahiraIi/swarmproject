"""
Main Application: Intelligent 2D Swarm Robotics Simulation
Entry point for the complete swarm simulation with ML integration
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# Import our custom modules
from swarm_engine import SwarmEnvironment
from visualization import SwarmVisualizer
from ml_agent import ObstacleRL_Agent, SwarmMLTrainer

def run_basic_simulation():
    """Run the basic interactive swarm simulation"""
    print("🤖 Starting Interactive Swarm Simulation...")
    print("Use mouse and keyboard to interact with the swarm!")
    print("Left click: Add obstacles | Right click: Select agent")
    print("Space: Pause | R: Reset | V: Toggle velocity vectors")
    
    try:
        visualizer = SwarmVisualizer()
        visualizer.run()
    except ImportError as e:
        print(f"❌ Error: Missing dependencies for visualization: {e}")
        print("Please install pygame: pip install pygame")
        return False
    except Exception as e:
        print(f"❌ Simulation error: {e}")
        return False
    
    return True

def run_ml_training(episodes: int = 500, save_model: bool = True):
    """Train the reinforcement learning agent"""
    print(f"🧠 Training ML Agent for {episodes} episodes...")
    print("The agent will learn to optimally challenge the swarm with obstacles.")
    
    try:
        # Create environment and agent
        swarm_env = SwarmEnvironment(800, 600, 15)
        state_size = 8  # Size of state vector from environment
        action_size = 21  # 20 positions + no-action
        
        agent = ObstacleRL_Agent(state_size, action_size)
        trainer = SwarmMLTrainer(swarm_env, agent)
        
        # Train the agent
        results = trainer.train(episodes=episodes, verbose=True)
        
        # Save results
        if save_model:
            agent.save_model("trained_obstacle_agent.pth")
            print("✅ Model saved as 'trained_obstacle_agent.pth'")
        
        # Plot training results
        plot_training_results(results)
        
        return True
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False

def run_ml_demo(model_path: str = "trained_obstacle_agent.pth"):
    """Run demonstration with trained ML agent"""
    print("🎯 Running ML Agent Demonstration...")
    
    try:
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            print("Please train a model first using: python main.py --mode train")
            return False
        
        # Create environment and load trained agent
        swarm_env = SwarmEnvironment(800, 600, 15)
        agent = ObstacleRL_Agent(8, 21)
        agent.load_model(model_path)
        agent.epsilon = 0.0  # No exploration during demo
        
        print("✅ Loaded trained agent")
        print("Watch as the AI strategically places obstacles to challenge the swarm!")
        
        # Run demonstration with visualization
        visualizer = SwarmVisualizer()
        visualizer.swarm_env = swarm_env
        
        # Add ML agent control to visualizer
        def ml_update():
            if len(visualizer.swarm_env.agents) > 0:
                state = visualizer.swarm_env.get_state_vector()
                if len(visualizer.swarm_env.obstacles) < 5:  # Max obstacles
                    action = agent.act(state, training=False)
                    if action < 20:  # Valid placement action
                        # Map action to position
                        grid_x = np.linspace(100, visualizer.sim_width - 100, 4)
                        grid_y = np.linspace(100, visualizer.sim_height - 100, 5)
                        positions = [(x, y) for x in grid_x for y in grid_y]
                        if action < len(positions):
                            x, y = positions[action]
                            visualizer.swarm_env.add_obstacle(x, y, 20)
        
        # Override update to include ML agent
        original_update = visualizer._update_simulation
        def enhanced_update():
            original_update()
            if np.random.random() < 0.05:  # 5% chance per frame
                ml_update()
        
        visualizer._update_simulation = enhanced_update
        visualizer.run()
        
        return True
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        return False

def run_analysis():
    """Run performance analysis and generate research plots"""
    print("📊 Running Swarm Performance Analysis...")
    
    try:
        # Test different swarm sizes
        swarm_sizes = [10, 15, 20, 25, 30]
        cohesion_scores = []
        speed_scores = []
        
        for size in swarm_sizes:
            print(f"Testing swarm size: {size}")
            env = SwarmEnvironment(800, 600, size)
            
            # Add some obstacles for challenge
            env.add_obstacle(200, 200, 30)
            env.add_obstacle(600, 400, 25)
            env.add_obstacle(400, 300, 35)
            
            # Run simulation
            cohesion_values = []
            speed_values = []
            
            for _ in range(500):  # Run for 500 steps
                env.update()
                cohesion_values.append(env.swarm_cohesion)
                speed_values.append(env.average_speed)
            
            # Average performance over last 200 steps (steady state)
            cohesion_scores.append(np.mean(cohesion_values[-200:]))
            speed_scores.append(np.mean(speed_values[-200:]))
        
        # Create analysis plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(swarm_sizes, cohesion_scores, 'b-o', linewidth=2)
        ax1.set_xlabel('Swarm Size')
        ax1.set_ylabel('Average Cohesion')
        ax1.set_title('Swarm Cohesion vs Size')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(swarm_sizes, speed_scores, 'r-o', linewidth=2)
        ax2.set_xlabel('Swarm Size')
        ax2.set_ylabel('Average Speed')
        ax2.set_title('Swarm Speed vs Size')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('swarm_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Analysis complete! Results saved as 'swarm_analysis.png'")
        
        # Print summary
        print("\n📈 Performance Summary:")
        for i, size in enumerate(swarm_sizes):
            print(f"Size {size:2d}: Cohesion = {cohesion_scores[i]:.3f}, Speed = {speed_scores[i]:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return False

def plot_training_results(results):
    """Plot ML training results"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Episode rewards
        ax1.plot(results['episode_rewards'], alpha=0.7)
        ax1.plot([np.mean(results['episode_rewards'][max(0, i-50):i+1]) 
                 for i in range(len(results['episode_rewards']))], 'r-', linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Progress')
        ax1.grid(True, alpha=0.3)
        
        # Training metrics
        if results['training_progress']:
            episodes = [p['episode'] for p in results['training_progress']]
            avg_rewards = [p['avg_reward'] for p in results['training_progress']]
            epsilons = [p['epsilon'] for p in results['training_progress']]
            
            ax2.plot(episodes, avg_rewards, 'b-', label='Avg Reward')
            ax2_twin = ax2.twinx()
            ax2_twin.plot(episodes, epsilons, 'g-', label='Epsilon')
            
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Average Reward', color='b')
            ax2_twin.set_ylabel('Epsilon', color='g')
            ax2.set_title('Learning Progress')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Training results saved as 'training_results.png'")
        
    except Exception as e:
        print(f"⚠️  Could not plot results: {e}")

def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║           🤖 Intelligent 2D Swarm Robotics Simulation       ║
    ║                                                              ║
    ║  🎯 Decentralized swarm intelligence                         ║
    ║  🧠 Machine learning integration                             ║
    ║  📊 Research-ready analysis tools                            ║
    ║  🎮 Interactive visualization                                ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Main application entry point"""
    print_banner()
    
    parser = argparse.ArgumentParser(
        description="Intelligent 2D Swarm Robotics Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode basic                    # Interactive simulation
  python main.py --mode train --episodes 1000   # Train ML agent
  python main.py --mode demo                     # ML agent demonstration
  python main.py --mode analysis                 # Performance analysis

For research applications, this simulation demonstrates:
- Decentralized swarm control algorithms
- Emergent collective behaviors  
- Machine learning integration with robotics
- Performance analysis and optimization
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['basic', 'train', 'demo', 'analysis'],
        default='basic',
        help='Simulation mode (default: basic)'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=500,
        help='Number of training episodes for ML mode (default: 500)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='trained_obstacle_agent.pth',
        help='Path to trained model file (default: trained_obstacle_agent.pth)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save trained model'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    try:
        import pygame
        import torch
        import matplotlib
    except ImportError as e:
        print(f"❌ Missing required dependencies: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    
    # Run selected mode
    success = False
    
    if args.mode == 'basic':
        success = run_basic_simulation()
    elif args.mode == 'train':
        success = run_ml_training(args.episodes, not args.no_save)
    elif args.mode == 'demo':
        success = run_ml_demo(args.model)
    elif args.mode == 'analysis':
        success = run_analysis()
    
    if success:
        print("\n✅ Simulation completed successfully!")
        print("\n🔬 Research Applications:")
        print("   • Demonstrates decentralized swarm control")
        print("   • Shows emergent collective behaviors")
        print("   • Integrates machine learning with robotics")
        print("   • Provides quantitative performance metrics")
        print("\n🎯 Perfect for showcasing skills in:")
        print("   • Autonomous systems")
        print("   • Multi-agent coordination")
        print("   • Machine learning in robotics")
        print("   • Research methodology")
    else:
        print("\n❌ Simulation encountered errors.")
        sys.exit(1)

if __name__ == "__main__":
    main() 