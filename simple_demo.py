"""
Simple Command-Line Swarm Demo
Shows core swarm algorithms without graphics
"""

import sys
import numpy as np
from swarm_engine import SwarmEnvironment

def print_banner():
    print("="*60)
    print("SWARM ROBOTICS SIMULATION")
    print("Decentralized Swarm Intelligence Demo")
    print("="*60)

def demonstrate_swarm_behavior():
    print("\nSWARM BEHAVIOR TEST")
    print("-" * 40)
    
    # create environment
    env = SwarmEnvironment(800, 600, 15)
    print(f"Created swarm with {len(env.agents)} agents")
    
    # add some obstacles
    env.add_obstacle(200, 200, 30)
    env.add_obstacle(600, 400, 25)
    env.add_obstacle(400, 300, 35)
    print(f"Added {len(env.obstacles)} obstacles")
    
    print("\nRunning simulation...")
    
    metrics_history = []
    
    for step in range(200):
        env.update()
        
        if step % 25 == 0:
            metrics = {
                'step': step,
                'cohesion': env.swarm_cohesion,
                'speed': env.average_speed,
                'collisions': env.total_collisions
            }
            metrics_history.append(metrics)
            
            print(f"Step {step:3d}: Cohesion={metrics['cohesion']:.3f}, "
                  f"Speed={metrics['speed']:.2f}, "
                  f"Collisions={metrics['collisions']}")
    
    print("\nPERFORMANCE RESULTS")
    print("-" * 40)
    
    # final metrics
    final_cohesion = np.mean([m['cohesion'] for m in metrics_history[-4:]])
    final_speed = np.mean([m['speed'] for m in metrics_history[-4:]])
    total_collisions = metrics_history[-1]['collisions']
    
    print(f"Final Performance:")
    print(f"   Average Cohesion: {final_cohesion:.3f}")
    print(f"   Average Speed:    {final_speed:.2f}")
    print(f"   Total Collisions: {total_collisions}")
    print(f"   Efficiency Score: {final_cohesion * final_speed:.3f}")
    
    print(f"\nAlgorithm Implementation:")
    print("   - Separation: Agents maintain spacing")
    print("   - Alignment: Coordinated movement")
    print("   - Cohesion: Group stays together")
    print("   - Obstacle Avoidance: Path planning")
    
    return metrics_history

def compare_swarm_sizes():
    print("\nSCALABILITY TEST")
    print("-" * 40)
    
    sizes = [5, 10, 15, 20, 25]
    results = []
    
    for size in sizes:
        print(f"Testing swarm size: {size}")
        env = SwarmEnvironment(800, 600, size)
        
        env.add_obstacle(200, 200, 30)
        env.add_obstacle(600, 400, 25)
        
        cohesion_vals = []
        speed_vals = []
        
        for step in range(100):
            env.update()
            if step > 20:
                cohesion_vals.append(env.swarm_cohesion)
                speed_vals.append(env.average_speed)
        
        avg_cohesion = np.mean(cohesion_vals)
        avg_speed = np.mean(speed_vals)
        efficiency = avg_cohesion * avg_speed
        
        results.append({
            'size': size,
            'cohesion': avg_cohesion,
            'speed': avg_speed,
            'efficiency': efficiency
        })
        
        print(f"   Size {size:2d}: Efficiency = {efficiency:.3f}")
    
    best_result = max(results, key=lambda x: x['efficiency'])
    print(f"\nOptimal Configuration:")
    print(f"   Best swarm size: {best_result['size']} agents")
    print(f"   Peak efficiency: {best_result['efficiency']:.3f}")
    
    return results

def demonstrate_research_relevance():
    print(f"\nRESEARCH APPLICATIONS")
    print("-" * 40)
    print("This simulation demonstrates:")
    print("   - Autonomous drone swarm coordination")
    print("   - Multi-robot navigation systems")
    print("   - Distributed control algorithms")
    print("   - Emergent behavior analysis")
    
    print(f"\nTECHNICAL FEATURES")
    print("-" * 40)
    print("Implementation includes:")
    print("   - Modular, object-oriented design")
    print("   - Performance metrics and analysis")
    print("   - Configurable parameters")
    print("   - Research-ready architecture")
    
    print(f"\nSKILLS DEMONSTRATED")
    print("-" * 40)
    print("Relevant to swarm robotics research:")
    print("   - Understanding of decentralized control")
    print("   - Multi-agent system design")
    print("   - Algorithm implementation")
    print("   - Performance analysis and metrics")
    print("   - Python programming")

def main():
    print_banner()
    
    try:
        print("\nStarting demonstration...")
        metrics = demonstrate_swarm_behavior()
        
        results = compare_swarm_sizes()
        
        demonstrate_research_relevance()
        
        print(f"\nDEMONSTRATION COMPLETE")
        print("="*60)
        print("Successfully demonstrated:")
        print("   - Decentralized swarm intelligence")
        print("   - Real-time performance metrics")
        print("   - Scalability analysis")
        print("   - Research-ready implementation")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 