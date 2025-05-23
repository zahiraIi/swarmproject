# Swarm Robotics Simulation

python-based sim focusing on reinforcement learning

## Core Functionality

- **decentralized control**: Agents operate based on local rules, without a central coordinator.
- **emergent behaviour**: Complex group patterns arise from simple agent interactions.
- **real-time visualization**: Interactive simulation with adjustable parameters.
- **mL integration**: Utilizes deep Q-Learning for tasks like strategic obstacle placement.
- **Performance Analysis**: Offers metrics and statistical analysis tools.

Key agent behaviors:
- **separation**: Avoid collisions with nearby agents.
- **alignment**: Steer towards the average heading of local flockmates.
- **cohesion**: Move towards the average position of local flockmates.
- **obstacle avoidance**: Navigate around static obstacles.
- **target seeking**: Move towards a designated target location.

## Quick Start

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd swarmproject
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run a demo:**
    ```bash
    # Interactive demo with visualization
    python demo.py

    # Command-line only demo (no graphics)
    python simple_demo.py

    # Reinforcement learning game demo
    python rl_game.py
    ```
    *Note: `main.py` is also available for more specific configurations, including training ML models.*

## Project Structure

-   `swarm_engine.py`: Defines core swarm behaviors (separation, alignment, cohesion, obstacle avoidance) and agent properties. Manages the simulation environment.
-   `ml_agent.py`: Implements the machine learning agent using Deep Q-Learning (DQN). Handles model training, prediction, and memory replay.
-   `visualization.py`: Provides the Pygame-based graphical interface for visualizing the swarm, obstacles, and agent states. Allows for real-time parameter adjustments.
-   `demo.py`: An interactive demonstration showcasing the swarm's capabilities with a visual interface.
-   `simple_demo.py`: A lightweight, command-line version of the demo, useful for environments without graphical support or for quick tests.
-   `rl_game.py`: A specific demo/game scenario where agents use reinforcement learning to navigate to a target while avoiding obstacles.
-   `main.py`: The main script for running simulations, including options for training ML models or running specific simulation configurations.
-   `analyze_swarm.py`: Contains tools for performance analysis, data collection, and statistical evaluation of swarm behavior. (Note: May require updates for current project state).
-   `run_demo.py`: A simple script to launch `demo.py`.
-   `requirements.txt`: Lists the Python package dependencies for the project.
-   `README.md`: This file. Provides an overview of the project.
-   `SHOWCASE_GUIDE.md`: Guide for showcasing project features. (Potentially outdated)
-   `RL_DEMO_README.md`: README specific to the RL demo. (Potentially outdated)


## Requirements

-   Python 3.8+
-   NumPy
-   Pygame
-   Matplotlib
-   PyTorch (for ML features)
-   SciPy (for analysis features)

Refer to `requirements.txt` for the full list of dependencies and versions.

## How it Works

The simulation is built around the `SwarmEnvironment` class in `swarm_engine.py`. Agents, also defined in `swarm_engine.py`, follow a set of rules (boids algorithm variants) to navigate. `ml_agent.py` introduces a learning component, allowing agents to adapt their behavior based on rewards and punishments. `visualization.py` uses Pygame to draw the simulation. Different scripts like `demo.py`, `simple_demo.py`, and `rl_game.py` provide ways to run and interact with the simulation.

## Research Applications

This simulation demonstrates concepts relevant to:
- Autonomous drone coordination
- Multi-robot navigation systems
- Distributed control algorithms
- Swarm intelligence research
- Real-time optimization

## Usage Examples

### Basic Simulation
```python
from swarm_engine import SwarmEnvironment

env = SwarmEnvironment(800, 600, 20)
env.add_obstacle(400, 300, 50)

for step in range(1000):
    env.update()
    print(f"Cohesion: {env.swarm_cohesion:.3f}")
```

### Analysis
```python
from analyze_swarm import run_performance_analysis

results = run_performance_analysis(
    swarm_sizes=[5, 10, 15, 20],
    num_trials=10
)
```

## Technical Details

**Algorithm**: Reynolds' boids with obstacle avoidance
**Physics**: Velocity-based movement with force limits
**ML**: PyTorch-based Deep Q-Learning
**Visualization**: Pygame with real-time controls
**Analysis**: Statistical testing with scipy

The implementation focuses on modularity and research applicability rather than performance optimization.

## Future Extensions

- 3D simulation environment
- ROS integration for real robots
- Advanced ML algorithms (PPO, A3C)
- Multi-swarm coordination
- Dynamic obstacle environments
