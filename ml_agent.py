"""
Machine Learning Agent: Reinforcement Learning for Intelligent Obstacle Placement
Demonstrates ML integration with swarm systems - learns to challenge swarm optimally
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import Tuple, List
import gymnasium as gym
from gymnasium import spaces

class DQNNetwork(nn.Module):
    """Deep Q-Network for learning optimal obstacle placement strategies"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
    def forward(self, x):
        return self.network(x)

class ObstacleRL_Agent:
    """RL Agent that learns to place obstacles strategically to challenge the swarm"""
    
    def __init__(self, state_size: int, action_size: int, lr: float = 1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Training parameters
        self.batch_size = 32
        self.gamma = 0.95  # discount factor
        self.tau = 1e-3    # soft update parameter
        self.update_every = 4
        self.step_count = 0
        
        # Performance tracking
        self.episode_rewards = []
        self.loss_history = []
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        if training and random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.loss_history.append(loss.item())
        
        # Soft update target network
        for target_param, local_param in zip(self.target_network.parameters(), 
                                           self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + 
                                  (1.0 - self.tau) * target_param.data)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath: str):
        """Save trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'loss_history': self.loss_history
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_rewards = checkpoint['episode_rewards']
        self.loss_history = checkpoint['loss_history']

class SwarmChallengeEnvironment:
    """RL Environment where agent learns to optimally challenge swarm with obstacles"""
    
    def __init__(self, swarm_env, max_obstacles: int = 5):
        self.swarm_env = swarm_env
        self.max_obstacles = max_obstacles
        self.action_space_size = 21  # 20 placement positions + 1 no-action
        
        # Grid for obstacle placement (4x5 grid)
        self.grid_x = np.linspace(100, swarm_env.width - 100, 4)
        self.grid_y = np.linspace(100, swarm_env.height - 100, 5)
        self.placement_positions = [(x, y) for x in self.grid_x for y in self.grid_y]
        
        # Episode tracking
        self.episode_step = 0
        self.max_episode_steps = 200
        self.initial_swarm_metrics = None
        
    def reset(self) -> np.ndarray:
        """Reset environment for new episode"""
        self.swarm_env.remove_obstacles()
        self.episode_step = 0
        
        # Let swarm stabilize without obstacles
        for _ in range(50):
            self.swarm_env.update()
        
        # Record initial performance
        self.initial_swarm_metrics = {
            'cohesion': self.swarm_env.swarm_cohesion,
            'speed': self.swarm_env.average_speed,
            'collisions': self.swarm_env.total_collisions
        }
        
        return self.swarm_env.get_state_vector()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action and return next state, reward, done, info"""
        reward = 0.0
        
        # Execute action (place obstacle or do nothing)
        if action < len(self.placement_positions) and len(self.swarm_env.obstacles) < self.max_obstacles:
            x, y = self.placement_positions[action]
            # Check if position is already occupied
            occupied = any(abs(ox - x) < 30 and abs(oy - y) < 30 
                          for ox, oy, _ in self.swarm_env.obstacles)
            if not occupied:
                self.swarm_env.add_obstacle(x, y, 20)
                reward += 0.1  # Small reward for placing obstacle
        
        # Update swarm for several steps
        prev_metrics = {
            'cohesion': self.swarm_env.swarm_cohesion,
            'speed': self.swarm_env.average_speed,
            'collisions': self.swarm_env.total_collisions
        }
        
        for _ in range(10):
            self.swarm_env.update()
        
        # Calculate reward based on swarm performance change
        reward += self._calculate_reward(prev_metrics)
        
        self.episode_step += 1
        done = self.episode_step >= self.max_episode_steps
        
        info = {
            'swarm_cohesion': self.swarm_env.swarm_cohesion,
            'average_speed': self.swarm_env.average_speed,
            'total_collisions': self.swarm_env.total_collisions,
            'num_obstacles': len(self.swarm_env.obstacles)
        }
        
        return self.swarm_env.get_state_vector(), reward, done, info
    
    def _calculate_reward(self, prev_metrics: dict) -> float:
        """Calculate reward based on how well agent challenges the swarm"""
        reward = 0.0
        
        # Reward for creating appropriate challenge
        current_cohesion = self.swarm_env.swarm_cohesion
        current_speed = self.swarm_env.average_speed
        current_collisions = self.swarm_env.total_collisions
        
        # Positive reward for reducing cohesion (forcing swarm to adapt)
        cohesion_change = prev_metrics['cohesion'] - current_cohesion
        reward += cohesion_change * 2.0
        
        # Moderate reward for slight speed reduction (shows challenge)
        speed_change = prev_metrics['speed'] - current_speed
        if 0 < speed_change < 0.5:  # Sweet spot
            reward += speed_change * 1.0
        elif speed_change > 0.5:  # Too much disruption
            reward -= 0.5
        
        # Small penalty for collisions (want challenge, not destruction)
        collision_increase = current_collisions - prev_metrics['collisions']
        reward -= collision_increase * 0.1
        
        # Bonus for maintaining swarm integrity while challenging
        if current_cohesion > 0.3 and current_speed > 1.0:
            reward += 0.2
        
        return reward

class SwarmMLTrainer:
    """Trainer class for the RL agent"""
    
    def __init__(self, swarm_env, agent: ObstacleRL_Agent):
        self.swarm_env = swarm_env
        self.agent = agent
        self.challenge_env = SwarmChallengeEnvironment(swarm_env)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_progress = []
        
    def train(self, episodes: int = 500, verbose: bool = True) -> dict:
        """Train the RL agent"""
        for episode in range(episodes):
            state = self.challenge_env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action = self.agent.act(state, training=True)
                next_state, reward, done, info = self.challenge_env.step(action)
                
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
                
                # Train the agent
                if len(self.agent.memory) > self.agent.batch_size:
                    self.agent.replay()
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.agent.episode_rewards.append(episode_reward)
            
            # Progress tracking
            if episode % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                self.training_progress.append({
                    'episode': episode,
                    'avg_reward': avg_reward,
                    'epsilon': self.agent.epsilon,
                    'avg_loss': np.mean(self.agent.loss_history[-100:]) if self.agent.loss_history else 0
                })
                
                if verbose:
                    print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                          f"Epsilon: {self.agent.epsilon:.3f}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'training_progress': self.training_progress,
            'final_epsilon': self.agent.epsilon
        } 