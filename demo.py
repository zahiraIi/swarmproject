"""
Simple Demo: Interactive 2D Swarm Robotics Simulation
Standalone demonstration without ML dependencies for easy setup
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

# Import our core modules
from swarm_engine import SwarmEnvironment
from visualization import SwarmVisualizer

def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║           🤖 Intelligent 2D Swarm Robotics Simulation       ║
    ║                     DEMONSTRATION VERSION                    ║
    ║                                                              ║
    ║  🎯 Decentralized swarm intelligence                         ║
    ║  🎮 Interactive visualization                                ║
    ║  📊 Real-time performance metrics                            ║
    ║  🔬 Research-ready architecture                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def run_interactive_simulation():
    """Run the basic interactive swarm simulation"""
    print("🤖 Starting Interactive Swarm Simulation...")
    print("Use mouse and keyboard to interact with the swarm!")
    print("")
    print("📋 Controls:")
    print("   • Left Click: Add obstacles")
    print("   • Right Click: Select agent") 
    print("   • Space: Play/Pause")
    print("   • R: Reset swarm")
    print("   • C: Clear obstacles")
    print("   • V: Toggle velocity vectors")
    print("   • G: Toggle grid")
    print("")
    print("🔬 What to observe:")
    print("   • Flocking behavior (separation, alignment, cohesion)")
    print("   • Obstacle avoidance")
    print("   • Emergent swarm intelligence")
    print("   • Real-time performance metrics")
    print("")
    
    try:
        visualizer = SwarmVisualizer()
        visualizer.run()
        return True
    except ImportError as e:
        print(f"❌ Error: Missing dependencies: {e}")
        print("Please install pygame: pip3 install pygame")
        return False
    except Exception as e:
        print(f"❌ Simulation error: {e}")
        return False

def run_performance_analysis():
    """Run a simple performance analysis"""
    print("📊 Running Swarm Performance Analysis...")
    
    try:
        # Test different swarm sizes
        swarm_sizes = [10, 15, 20, 25, 30]
        cohesion_scores = []
        speed_scores = []
        
        for size in swarm_sizes:
            print(f"   Testing swarm size: {size}")
            env = SwarmEnvironment(800, 600, size)
            
            # Add some obstacles for challenge
            env.add_obstacle(200, 200, 30)
            env.add_obstacle(600, 400, 25)
            env.add_obstacle(400, 300, 35)
            
            # Run simulation
            cohesion_values = []
            speed_values = []
            
            for _ in range(300):  # Run for 300 steps
                env.update()
                if _ > 50:  # Skip initial settling
                    cohesion_values.append(env.swarm_cohesion)
                    speed_values.append(env.average_speed)
            
            # Average performance
            cohesion_scores.append(np.mean(cohesion_values))
            speed_scores.append(np.mean(speed_values))
        
        # Create analysis plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(swarm_sizes, cohesion_scores, 'b-o', linewidth=2, markersize=8)
        ax1.set_xlabel('Swarm Size')
        ax1.set_ylabel('Average Cohesion')
        ax1.set_title('Swarm Cohesion vs Size')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(swarm_sizes, speed_scores, 'r-o', linewidth=2, markersize=8)
        ax2.set_xlabel('Swarm Size')
        ax2.set_ylabel('Average Speed')
        ax2.set_title('Swarm Speed vs Size')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('swarm_demo_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Analysis complete! Results saved as 'swarm_demo_analysis.png'")
        
        # Print summary
        print("\n📈 Performance Summary:")
        for i, size in enumerate(swarm_sizes):
            print(f"   Size {size:2d}: Cohesion = {cohesion_scores[i]:.3f}, Speed = {speed_scores[i]:.2f}")
        
        # Find optimal size
        efficiency_scores = [c * s for c, s in zip(cohesion_scores, speed_scores)]
        optimal_idx = np.argmax(efficiency_scores)
        print(f"\n🎯 Optimal swarm size: {swarm_sizes[optimal_idx]} agents")
        print(f"   (Efficiency score: {efficiency_scores[optimal_idx]:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return False

def demonstrate_algorithms():
    """Demonstrate different swarm behaviors"""
    print("🧠 Demonstrating Swarm Intelligence Algorithms...")
    print("This simulation implements the classic 'Boids' algorithm:")
    print("")
    print("1. 🔄 SEPARATION: Agents avoid crowding neighbors")
    print("   - Each agent maintains minimum distance from others")
    print("   - Prevents collisions and clustering")
    print("")
    print("2. ➡️  ALIGNMENT: Agents align with neighbor velocities")  
    print("   - Creates coordinated group movement")
    print("   - Enables flocking behavior")
    print("")
    print("3. 🎯 COHESION: Agents move toward neighbor center")
    print("   - Keeps the group together")
    print("   - Prevents fragmentation")
    print("")
    print("4. 🚧 OBSTACLE AVOIDANCE: Agents navigate around barriers")
    print("   - Dynamic path planning")
    print("   - Collision prevention")
    print("")
    print("🔬 Research Applications:")
    print("   • Drone swarm coordination")
    print("   • Robot navigation systems") 
    print("   • Crowd simulation")
    print("   • Biological modeling")
    print("   • Autonomous vehicle coordination")

def main():
    """Main demo application"""
    print_banner()
    
    parser = argparse.ArgumentParser(
        description="Swarm Robotics Simulation Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 demo.py                        # Interactive simulation
  python3 demo.py --mode analysis        # Performance analysis
  python3 demo.py --mode info            # Algorithm information

Perfect for demonstrating understanding of:
- Decentralized swarm control algorithms
- Emergent collective behaviors
- Real-time simulation and visualization
- Performance analysis and optimization
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['interactive', 'analysis', 'info'],
        default='interactive',
        help='Demo mode (default: interactive)'
    )
    
    args = parser.parse_args()
    
    # Check basic dependencies
    try:
        import pygame
        import matplotlib
    except ImportError as e:
        print(f"❌ Missing required dependencies: {e}")
        print("Please install: pip3 install pygame matplotlib numpy")
        sys.exit(1)
    
    # Run selected mode
    success = False
    
    if args.mode == 'interactive':
        success = run_interactive_simulation()
    elif args.mode == 'analysis':
        success = run_performance_analysis()
    elif args.mode == 'info':
        demonstrate_algorithms()
        success = True
    
    if success:
        print("\n✅ Demo completed successfully!")
        print("\n🎯 Key Demonstrations:")
        print("   • Decentralized swarm control (no central coordinator)")
        print("   • Emergent collective behaviors from simple rules")
        print("   • Real-time interactive simulation")
        print("   • Performance analysis and optimization")
        print("   • Research-ready modular architecture")
        print("")
        print("🔬 Perfect for showcasing skills in:")
        print("   • Autonomous systems and robotics")
        print("   • Multi-agent coordination algorithms")
        print("   • Real-time simulation and visualization")
        print("   • Research methodology and analysis")
        print("   • Software engineering best practices")
    else:
        print("\n❌ Demo encountered errors.")
        sys.exit(1)

if __name__ == "__main__":
    main() 