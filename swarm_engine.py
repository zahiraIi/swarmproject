import numpy as np
import math
from typing import List, Tuple, Optional

class Agent:
    
    def __init__(self, x: float, y: float, world_width: int, world_height: int):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.random.uniform(-2, 2, 2)
        self.acceleration = np.zeros(2)
        
        # basic physics stuff
        self.max_speed = 3.0
        self.max_force = 0.1
        self.world_width = world_width
        self.world_height = world_height
        
        # behavior settings
        self.neighbor_radius = 50.0
        self.separation_radius = 25.0
        
        # tracking
        self.collision_count = 0
        self.distance_traveled = 0.0
        self.last_position = self.position.copy()
        
    def update(self, agents: List['Agent'], obstacles: List[Tuple[float, float, float]]) -> None:
        
        # get forces from swarm behaviors
        separation_force = self._separation(agents)
        alignment_force = self._alignment(agents)
        cohesion_force = self._cohesion(agents)
        obstacle_force = self._avoid_obstacles(obstacles)
        
        # combine with weights
        self.acceleration = (
            2.0 * separation_force +
            1.0 * alignment_force + 
            1.0 * cohesion_force +
            3.0 * obstacle_force
        )
        
        # update position
        self.velocity += self.acceleration
        self.velocity = self._limit_magnitude(self.velocity, self.max_speed)
        
        self.distance_traveled += np.linalg.norm(self.velocity)
        
        self.position += self.velocity
        self._wrap_around()
        
        self.acceleration *= 0
        
    def _separation(self, agents: List['Agent']) -> np.ndarray:
        """avoid crowding neighbors"""
        steer = np.zeros(2)
        count = 0
        
        for agent in agents:
            distance = np.linalg.norm(self.position - agent.position)
            if 0 < distance < self.separation_radius:
                diff = self.position - agent.position
                diff = diff / distance  
                steer += diff
                count += 1
                
        if count > 0:
            steer /= count
            steer = self._limit_magnitude(steer, self.max_force)
            
        return steer
    
    def _alignment(self, agents: List['Agent']) -> np.ndarray:
        """steer towards average heading of neighbors"""
        avg_velocity = np.zeros(2)
        count = 0
        
        for agent in agents:
            distance = np.linalg.norm(self.position - agent.position)
            if 0 < distance < self.neighbor_radius:
                avg_velocity += agent.velocity
                count += 1
                
        if count > 0:
            avg_velocity /= count
            avg_velocity = self._limit_magnitude(avg_velocity, self.max_speed)
            steer = avg_velocity - self.velocity
            steer = self._limit_magnitude(steer, self.max_force)
            return steer
            
        return np.zeros(2)
    
    def _cohesion(self, agents: List['Agent']) -> np.ndarray:
        """steer towards center of neighbors"""
        center_of_mass = np.zeros(2)
        count = 0
        
        for agent in agents:
            distance = np.linalg.norm(self.position - agent.position)
            if 0 < distance < self.neighbor_radius:
                center_of_mass += agent.position
                count += 1
                
        if count > 0:
            center_of_mass /= count
            desired = center_of_mass - self.position
            desired = self._limit_magnitude(desired, self.max_speed)
            steer = desired - self.velocity
            steer = self._limit_magnitude(steer, self.max_force)
            return steer
            
        return np.zeros(2)
    
    def _avoid_obstacles(self, obstacles: List[Tuple[float, float, float]]) -> np.ndarray:
        """avoid hitting stuff"""
        steer = np.zeros(2)
        
        for ox, oy, radius in obstacles:
            obstacle_pos = np.array([ox, oy])
            distance = np.linalg.norm(self.position - obstacle_pos)
            
            if distance < radius + 5: 
                self.collision_count += 1
            
            if distance < radius + 50:  
                diff = self.position - obstacle_pos
                if distance > 0:
                    force_magnitude = max(0, (radius + 50 - distance) / 50)
                    diff = diff / distance * force_magnitude
                    steer += diff
                    
        return self._limit_magnitude(steer, self.max_force * 2)
    
    def _limit_magnitude(self, vector: np.ndarray, max_mag: float) -> np.ndarray:
        mag = np.linalg.norm(vector)
        if mag > max_mag:
            return vector / mag * max_mag
        return vector
    
    def _wrap_around(self) -> None:
        if self.position[0] < 0:
            self.position[0] = self.world_width
        elif self.position[0] > self.world_width:
            self.position[0] = 0
            
        if self.position[1] < 0:
            self.position[1] = self.world_height
        elif self.position[1] > self.world_height:
            self.position[1] = 0

class SwarmEnvironment:
    
    def __init__(self, width: int, height: int, num_agents: int):
        self.width = width
        self.height = height
        self.agents = []
        self.obstacles = []
        
        self.total_collisions = 0
        self.average_speed = 0.0
        self.swarm_cohesion = 0.0
        
        # spawn agents randomly
        for _ in range(num_agents):
            x = np.random.uniform(50, width - 50)
            y = np.random.uniform(50, height - 50)
            self.agents.append(Agent(x, y, width, height))
    
    def add_obstacle(self, x: float, y: float, radius: float) -> None:
        self.obstacles.append((x, y, radius))
    
    def remove_obstacles(self) -> None:
        self.obstacles.clear()
    
    def update(self) -> None:
        # update everyone
        for agent in self.agents:
            agent.update(self.agents, self.obstacles)
        
        self._calculate_metrics()
    
    def _calculate_metrics(self) -> None:
        if not self.agents:
            return
            
        self.total_collisions = sum(agent.collision_count for agent in self.agents)
        
        speeds = [np.linalg.norm(agent.velocity) for agent in self.agents]
        self.average_speed = np.mean(speeds)
        
        # cohesion = how close agents are to center
        positions = np.array([agent.position for agent in self.agents])
        center = np.mean(positions, axis=0)
        distances = [np.linalg.norm(pos - center) for pos in positions]
        self.swarm_cohesion = 1.0 / (1.0 + np.mean(distances))
    
    def get_state_vector(self) -> np.ndarray:
        """for ML stuff"""
        if not self.agents:
            return np.zeros(8)
            
        positions = np.array([agent.position for agent in self.agents])
        velocities = np.array([agent.velocity for agent in self.agents])
        
        center_of_mass = np.mean(positions, axis=0)
        avg_velocity = np.mean(velocities, axis=0)
        spread = np.std(positions, axis=0)
        
        state = np.concatenate([
            center_of_mass / [self.width, self.height],  
            avg_velocity / 3.0,  
            spread / [self.width, self.height],  
            [len(self.obstacles) / 10.0, self.average_speed / 3.0]  
        ])
        
        return state.astype(np.float32) 