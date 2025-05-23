import pygame
import numpy as np
import random
import time
from typing import List, Tuple, Optional

# Simple colors matching the example
COLORS = {
    'background': (240, 240, 240),  # Light gray
    'grid': (200, 200, 200),        # Gray grid lines
    'obstacle': (255, 100, 100),    # Red circles
    'agent': (100, 150, 255),       # Blue square/circle
    'target': (255, 200, 0),        # Yellow target
    'button': (220, 220, 220),      # Light gray buttons
    'text': (0, 0, 0),              # Black text
    'slider': (100, 100, 100),      # Slider track
    'slider_handle': (50, 50, 50)   # Slider handle
}

class SimpleAgent:
    MIN_EXPERIENCE_FOR_AVOIDANCE = 20  # Slightly earlier engagement of avoidance
    TOTAL_EXPERIENCE_FOR_MAX_LEARNING = 1000 
    MAX_COLLISION_PENALTIES_EFFECT = 15 # Max effect reached with fewer penalties
    HIT_DETECTION_RADIUS_OFFSET = 2     
    GENERAL_AWARENESS_RADIUS_OFFSET = 45 # Increased awareness radius

    def __init__(self, x: float, y: float):
        self.start_x = x  # Remember starting position
        self.start_y = y
        self.x = x
        self.y = y
        self.vx = random.uniform(-0.3, 0.3) # Start slower for clearer observation
        self.vy = random.uniform(-0.3, 0.3)
        self.learning_rate = 0.12
        self.experience = 0
        self.collision_penalty_score = 0 # Accumulated from resets
        self.successful_reaches = 0 # Track target reaches
        self.total_resets = 0 # Track how many times reset due to obstacles
        self.has_reached_destination = False  # track if agent reached destination
        
        # Anti-stuck mechanism
        self.last_significant_move_x = x
        self.last_significant_move_y = y
        self.stuck_timer = 0
        
    def update(self, target_x: float, target_y: float, obstacles: List[Tuple[float, float, float]], other_agents: List['SimpleAgent']):
        # If already at destination, stay there with slight random movement to show they're "alive"
        if self.has_reached_destination:
            # Small random movement to show activity
            self.x += random.uniform(-0.5, 0.5)
            self.y += random.uniform(-0.5, 0.5)
            # Keep within destination area
            dest_dist = np.sqrt((self.x - target_x)**2 + (self.y - target_y)**2)
            if dest_dist > 25:
                direction_x = (target_x - self.x) / dest_dist
                direction_y = (target_y - self.y) / dest_dist
                self.x = target_x - direction_x * 25
                self.y = target_y - direction_y * 25
            return
            
        target_dx = target_x - self.x
        target_dy = target_y - self.y
        target_dist = np.sqrt(target_dx**2 + target_dy**2)
        
        desired_vx, desired_vy = 0, 0
        avoid_vx, avoid_vy = 0, 0
        flock_vx, flock_vy = 0, 0
        
        # Check if reached target (success!)
        if target_dist < 30:  # Close enough to target
            self.successful_reaches += 1
            self.has_reached_destination = True  # Stay at destination
            return  # Skip rest of update this frame
        
        # --- Collective Swarm Behavior (Flocking) ---
        if len(other_agents) > 0:
            # Separation: avoid crowding neighbors
            sep_vx, sep_vy = 0, 0
            # Alignment: steer towards average heading of neighbors
            align_vx, align_vy = 0, 0
            # Cohesion: steer towards average position of neighbors
            coh_vx, coh_vy = 0, 0
            
            neighbor_count = 0
            neighbor_separation_count = 0
            
            for other in other_agents:
                if other == self or other.has_reached_destination:
                    continue
                    
                dx = self.x - other.x
                dy = self.y - other.y
                distance = np.sqrt(dx**2 + dy**2)
                
                # Only consider nearby agents (increased radius for better coordination)
                if distance < 100 and distance > 0:
                    neighbor_count += 1
                    
                    # Alignment - match velocity
                    align_vx += other.vx
                    align_vy += other.vy
                    
                    # Cohesion - move towards center (reduced for more spacing)
                    coh_vx += other.x
                    coh_vy += other.y
                    
                    # Separation - avoid crowding (increased distance and stronger effect)
                    if distance < 60:  # Increased from 40 to 60 for more spacing
                        neighbor_separation_count += 1
                        sep_force = 2.0 / (distance + 1)  # Increased from 1.0 to 2.0 for stronger separation
                        sep_vx += (dx / distance) * sep_force
                        sep_vy += (dy / distance) * sep_force
            
            if neighbor_count > 0:
                # Alignment
                align_vx /= neighbor_count
                align_vy /= neighbor_count
                flock_vx += align_vx * 0.3
                flock_vy += align_vy * 0.3
                
                # Cohesion - reduced strength for more spacing
                coh_vx = (coh_vx / neighbor_count) - self.x
                coh_vy = (coh_vy / neighbor_count) - self.y
                flock_vx += coh_vx * 0.05  # Reduced from 0.1 to 0.05
                flock_vy += coh_vy * 0.05  # Reduced from 0.1 to 0.05
            
            if neighbor_separation_count > 0:
                # Separation - increased strength for more spacing
                flock_vx += sep_vx * 0.8  # Increased from 0.5 to 0.8
                flock_vy += sep_vy * 0.8  # Increased from 0.5 to 0.8
        
        # --- Target Seeking Behavior (Original Direct Logic) ---
        if target_dist > 0: # target_dist is calculated at the beginning of update
            overall_learning_factor = min(self.experience / self.TOTAL_EXPERIENCE_FOR_MAX_LEARNING, 1.0)
            base_speed = 0.4 + 0.8 * overall_learning_factor # Gradually increase speed with experience
            
            # Directly use target_dx and target_dy (calculated from actual target_x, target_y)
            desired_vx = (target_dx / target_dist) * base_speed
            desired_vy = (target_dy / target_dist) * base_speed
        
        # --- Obstacle Collision Detection & Reset ---
        if self.experience > self.MIN_EXPERIENCE_FOR_AVOIDANCE:
            for ox, oy, radius in obstacles:
                dx_obs = self.x - ox
                dy_obs = self.y - oy
                dist_obs = np.sqrt(dx_obs**2 + dy_obs**2)

                # Hit detection - RESET TO ORIGIN on collision
                if dist_obs < radius + self.HIT_DETECTION_RADIUS_OFFSET:
                    # PUNISHMENT: Reset to starting position
                    self.x = self.start_x + random.uniform(-20, 20)
                    self.y = self.start_y + random.uniform(-20, 20)
                    self.vx = random.uniform(-0.3, 0.3)
                    self.vy = random.uniform(-0.3, 0.3)
                    self.has_reached_destination = False  # Reset destination status
                    
                    # Increase penalty score (learning from failure)
                    self.collision_penalty_score = min(self.collision_penalty_score + 1, self.MAX_COLLISION_PENALTIES_EFFECT)
                    self.total_resets += 1
                    
                    return  # Skip rest of movement this frame since we reset
                
                # Proactive avoidance ONLY IF agent has been penalized before
                elif self.collision_penalty_score > 0 and dist_obs < radius + self.GENERAL_AWARENESS_RADIUS_OFFSET:
                    # Avoidance strength based on how many times agent has been reset
                    # More resets = stronger avoidance, less dependent on general experience for base effect
                    penalty_factor = min(self.collision_penalty_score / self.MAX_COLLISION_PENALTIES_EFFECT, 1.0)
                    
                    # Increased base strength and stronger penalty scaling
                    avoidance_strength = (0.5 + penalty_factor * 1.0) * 2.0 # Base avoidance + learned penalty * multiplier
                    
                    if dist_obs > 0.1:
                        avoid_vx += (dx_obs / dist_obs) * avoidance_strength
                        avoid_vy += (dy_obs / dist_obs) * avoidance_strength
            
        # --- Movement ---
        self.vx = self.vx * 0.75 + (desired_vx + avoid_vx + flock_vx) * self.learning_rate
        self.vy = self.vy * 0.75 + (desired_vy + avoid_vy + flock_vy) * self.learning_rate
        
        # Speed limiting
        current_max_speed = 0.4 + 0.8 * min(self.experience / self.TOTAL_EXPERIENCE_FOR_MAX_LEARNING, 1.0)
        speed = np.sqrt(self.vx**2 + self.vy**2)
        if speed > current_max_speed:
            self.vx = (self.vx / speed) * current_max_speed
            self.vy = (self.vy / speed) * current_max_speed
        
        # Anti-stuck: Check if agent has moved significantly
        if np.sqrt((self.x - self.last_significant_move_x)**2 + (self.y - self.last_significant_move_y)**2) < 0.5: # Threshold for being stuck
            self.stuck_timer += 1
        else:
            self.last_significant_move_x = self.x
            self.last_significant_move_y = self.y
            self.stuck_timer = 0

        if self.stuck_timer > 90: # If stuck for ~3 seconds (90 frames at 30fps)
            # Apply a small random nudge
            self.vx += random.uniform(-0.2, 0.2)
            self.vy += random.uniform(-0.2, 0.2)
            self.stuck_timer = 0 # Reset timer after nudge
            
        self.x += self.vx
        self.y += self.vy
        
        # Boundary collision also resets (walls are also obstacles)
        if self.x < 10 or self.x > 790 or self.y < 10 or self.y > 590:
            self.x = self.start_x + random.uniform(-20, 20)
            self.y = self.start_y + random.uniform(-20, 20)
            self.vx = random.uniform(-0.3, 0.3)
            self.vy = random.uniform(-0.3, 0.3)
            self.has_reached_destination = False  # Reset destination status
            self.total_resets += 1
        
        self.experience += 1

class RLSwarmGame:
    def __init__(self):
        pygame.init()
        self.width = 1000
        self.height = 700
        self.sim_width = 800
        self.sim_height = 600
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RL Swarm Learning Demo")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
        # Game state
        self.running = True
        self.paused = False
        self.editing_obstacles = False
        self.mission_completed = False  # Track if all agents reached destination
        
        # Obstacle editing state
        self.dragging_obstacle = None  # Index of obstacle being dragged
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        
        # Speed control
        self.speed_multiplier = 1.0
        self.simulation_steps_to_run = 0
        
        # Swarm setup
        self.agents = []
        self.obstacles = []
        self.target_x = 700
        self.target_y = 100
        
        # Performance tracking
        self.steps = 0
        self.accuracy_history = []
        self.max_history = 100
        
        # Buttons
        button_y = 620
        self.buttons = {
            'start': pygame.Rect(50, button_y, 80, 40),
            'pause': pygame.Rect(150, button_y, 80, 40),
            'restart': pygame.Rect(250, button_y, 80, 40),
            'edit_obstacles': pygame.Rect(350, button_y, 120, 40)
        }
        
        # Speed slider
        self.speed_slider = {
            'rect': pygame.Rect(820, 320, 150, 20),
            'handle_x': 820 + 75, 
            'dragging': False,
            'min_speed': 0.1,
            'max_speed': 5.0
        }
        
        self.reset_simulation()
    
    def reset_simulation(self):
        self.agents = []
        for i in range(8): # Number of agents
            x = 100 + random.uniform(-30, 30)
            y = 300 + random.uniform(-30, 30)
            agent = SimpleAgent(x, y)
            agent.experience = 0
            agent.collision_penalty_score = 0
            agent.successful_reaches = 0  # Reset success tracking
            agent.total_resets = 0        # Reset failure tracking
            agent.has_reached_destination = False  # Reset destination status
            self.agents.append(agent)
        
        self.obstacles = [
            (300, 200, 40),
            (500, 350, 50),
            (650, 250, 45),
            (750, 450, 40)
        ]
        
        self.steps = 0 # Reset global steps counter for the new simulation
        self.accuracy_history = []
        self.mission_completed = False  # New: track if all agents reached destination
    
    def calculate_accuracy(self) -> float:
        if not self.agents: return 0
        
        # Calculate success rate: successful reaches vs total attempts (resets + successes)
        total_successes = sum(agent.successful_reaches for agent in self.agents)
        total_attempts = sum(agent.total_resets + agent.successful_reaches for agent in self.agents)
        
        if total_attempts == 0:
            return 0
        
        success_rate = (total_successes / total_attempts) * 100
        return min(success_rate, 100)  # Cap at 100%
    
    def get_learning_stats(self):
        """Get detailed learning statistics for display"""
        if not self.agents:
            return {"avg_resets": 0, "avg_successes": 0, "total_attempts": 0}
        
        total_resets = sum(agent.total_resets for agent in self.agents)
        total_successes = sum(agent.successful_reaches for agent in self.agents)
        avg_resets = total_resets / len(self.agents)
        avg_successes = total_successes / len(self.agents)
        total_attempts = total_resets + total_successes
        
        return {
            "avg_resets": avg_resets,
            "avg_successes": avg_successes, 
            "total_attempts": total_attempts,
            "success_rate": (total_successes / max(total_attempts, 1)) * 100
        }
    
    def update_speed_from_slider(self, mouse_x: int):
        slider_rect = self.speed_slider['rect']
        relative_x = max(0, min(slider_rect.width, mouse_x - slider_rect.x))
        ratio = relative_x / slider_rect.width
        min_speed, max_speed = self.speed_slider['min_speed'], self.speed_slider['max_speed']
        self.speed_multiplier = min_speed + ratio * (max_speed - min_speed)
        self.speed_slider['handle_x'] = slider_rect.x + relative_x
    
    def find_obstacle_at_position(self, x: int, y: int) -> int:
        """Find obstacle at given position, return index or -1 if none found"""
        for i, (ox, oy, radius) in enumerate(self.obstacles):
            distance = np.sqrt((ox - x)**2 + (oy - y)**2)
            if distance <= radius:
                return i
        return -1
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    pos = event.pos
                    if self.buttons['start'].collidepoint(pos): self.paused = False
                    elif self.buttons['pause'].collidepoint(pos): self.paused = True
                    elif self.buttons['restart'].collidepoint(pos): self.reset_simulation()
                    elif self.buttons['edit_obstacles'].collidepoint(pos): 
                        self.editing_obstacles = not self.editing_obstacles
                        self.dragging_obstacle = None  # Reset dragging state
                    elif self.speed_slider['rect'].collidepoint(pos):
                        self.speed_slider['dragging'] = True
                        self.update_speed_from_slider(pos[0])
                    elif self.editing_obstacles and pos[0] < self.sim_width and pos[1] < self.sim_height:
                        # Check if clicking on existing obstacle to drag it
                        obstacle_index = self.find_obstacle_at_position(pos[0], pos[1])
                        if obstacle_index >= 0:
                            # Start dragging existing obstacle
                            self.dragging_obstacle = obstacle_index
                            ox, oy, radius = self.obstacles[obstacle_index]
                            self.drag_offset_x = pos[0] - ox
                            self.drag_offset_y = pos[1] - oy
                        else:
                            # Add new obstacle
                            self.obstacles.append((pos[0], pos[1], random.uniform(25, 50)))
                elif event.button == 3 and self.editing_obstacles:
                    pos = event.pos
                    # Right-click: remove obstacle or resize if clicking on one
                    obstacle_index = self.find_obstacle_at_position(pos[0], pos[1])
                    if obstacle_index >= 0:
                        # If shift is held, resize instead of delete
                        keys = pygame.key.get_pressed()
                        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                            # Resize obstacle (increase radius)
                            ox, oy, radius = self.obstacles[obstacle_index]
                            new_radius = min(radius + 5, 80)  # Max radius of 80
                            self.obstacles[obstacle_index] = (ox, oy, new_radius)
                        else:
                            # Delete obstacle
                            self.obstacles.pop(obstacle_index)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: 
                    self.speed_slider['dragging'] = False
                    self.dragging_obstacle = None  # Stop dragging obstacle
            elif event.type == pygame.MOUSEMOTION:
                if self.speed_slider['dragging']: 
                    self.update_speed_from_slider(event.pos[0])
                elif self.dragging_obstacle is not None and self.editing_obstacles:
                    # Update obstacle position while dragging
                    new_x = event.pos[0] - self.drag_offset_x
                    new_y = event.pos[1] - self.drag_offset_y
                    # Keep obstacle within simulation bounds
                    new_x = max(25, min(self.sim_width - 25, new_x))
                    new_y = max(25, min(self.sim_height - 25, new_y))
                    
                    ox, oy, radius = self.obstacles[self.dragging_obstacle]
                    self.obstacles[self.dragging_obstacle] = (new_x, new_y, radius)
    
    def update(self):
        if not self.paused and not self.mission_completed:
            self.simulation_steps_to_run += self.speed_multiplier
            steps_this_frame = int(self.simulation_steps_to_run)
            self.simulation_steps_to_run -= steps_this_frame
            
            for _ in range(steps_this_frame):
                if not self.agents: continue
                for agent in self.agents:
                    agent.update(self.target_x, self.target_y, self.obstacles, self.agents)
                
                # Check if all agents have reached destination
                agents_at_destination = sum(1 for agent in self.agents if agent.has_reached_destination)
                if agents_at_destination == len(self.agents) and len(self.agents) > 0:
                    self.mission_completed = True
                    print(f"Mission completed! All {len(self.agents)} agents reached the destination after {self.steps} steps.")
                    break  # Stop simulation steps
                
                self.steps += 1 # This is the global step counter for the current simulation/generation
                
                if self.steps % 30 == 0:
                    accuracy = self.calculate_accuracy()
                    self.accuracy_history.append(accuracy)
                    if len(self.accuracy_history) > self.max_history: self.accuracy_history.pop(0)
    
    def draw(self):
        self.screen.fill(COLORS['background'])
        for x_grid in range(0, self.sim_width, 50):
            pygame.draw.line(self.screen, COLORS['grid'], (x_grid, 0), (x_grid, self.sim_height))
        for y_grid in range(0, self.sim_height, 50):
            pygame.draw.line(self.screen, COLORS['grid'], (0, y_grid), (self.sim_width, y_grid))
        pygame.draw.rect(self.screen, COLORS['grid'], (0, 0, self.sim_width, self.sim_height), 2)
        
        for ox, oy, radius in self.obstacles:
            pygame.draw.circle(self.screen, COLORS['obstacle'], (int(ox), int(oy)), int(radius))
        
        pygame.draw.circle(self.screen, COLORS['target'], (int(self.target_x), int(self.target_y)), 25)
        target_text = self.font.render("B", True, COLORS['text'])
        target_text_rect = target_text.get_rect(center=(int(self.target_x), int(self.target_y)))
        self.screen.blit(target_text, target_text_rect)
        
        for agent in self.agents:
            agent_rect = pygame.Rect(int(agent.x - 4), int(agent.y - 4), 8, 8)
            # Different color for agents that reached destination
            agent_color = (100, 255, 100) if agent.has_reached_destination else COLORS['agent']
            pygame.draw.rect(self.screen, agent_color, agent_rect)
        
        for name, rect in self.buttons.items():
            btn_color = COLORS['button']
            if name == 'edit_obstacles' and self.editing_obstacles: btn_color = (180, 255, 180)
            pygame.draw.rect(self.screen, btn_color, rect)
            pygame.draw.rect(self.screen, COLORS['text'], rect, 2)
            btn_text = self.font.render(name.replace('_', ' ').title(), True, COLORS['text'])
            btn_text_rect = btn_text.get_rect(center=rect.center)
            self.screen.blit(btn_text, btn_text_rect)
        
        slider_rect = self.speed_slider['rect']
        pygame.draw.rect(self.screen, COLORS['slider'], slider_rect)
        pygame.draw.rect(self.screen, COLORS['text'], slider_rect, 2)
        handle_x = int(self.speed_slider['handle_x'])
        handle_rect = pygame.Rect(handle_x - 5, slider_rect.y - 3, 10, slider_rect.height + 6)
        pygame.draw.rect(self.screen, COLORS['slider_handle'], handle_rect)
        speed_label_text = self.font.render("Speed:", True, COLORS['text'])
        self.screen.blit(speed_label_text, (820, 295))
        speed_value_text = self.font.render(f"{self.speed_multiplier:.1f}x", True, COLORS['text'])
        self.screen.blit(speed_value_text, (820, 345))
        
        if self.accuracy_history:
            current_accuracy = self.accuracy_history[-1]
            stats = self.get_learning_stats()
            
            # Show mission status
            mission_status = "MISSION COMPLETED!" if self.mission_completed else "In Progress"
            mission_color = (0, 255, 0) if self.mission_completed else COLORS['text']
            agents_at_dest = sum(1 for agent in self.agents if agent.has_reached_destination)
            
            progress_display = [
                f"Steps: {self.steps}",
                f"Mission: {mission_status}",
                f"Agents at Destination: {agents_at_dest}/{len(self.agents)}",
                f"Success Rate: {current_accuracy:.1f}%",
                f"Avg Resets: {stats['avg_resets']:.1f}",
                f"Avg Successes: {stats['avg_successes']:.1f}",
                f"Learning: {'█' * int(current_accuracy/10)}{'░' * (10-int(current_accuracy/10))}"
            ]
            for i, line in enumerate(progress_display):
                color = mission_color if i == 1 else COLORS['text']  # Highlight mission status
                line_surf = self.font.render(line, True, color)
                self.screen.blit(line_surf, (820, 50 + i * 25))  # Reduced spacing
            
            if len(self.accuracy_history) > 1:
                graph_ui_rect = pygame.Rect(820, 200, 150, 100)  # Moved up
                pygame.draw.rect(self.screen, (250, 250, 250), graph_ui_rect)
                pygame.draw.rect(self.screen, COLORS['text'], graph_ui_rect, 1)
                
                # Title for graph
                graph_title = pygame.font.Font(None, 18).render("Success Rate", True, COLORS['text'])
                self.screen.blit(graph_title, (820, 185))
                
                points = []
                num_points_for_graph = len(self.accuracy_history)
                if num_points_for_graph > 1:
                    for i_hist, acc_hist in enumerate(self.accuracy_history):
                        x_pt = graph_ui_rect.x + int(i_hist * graph_ui_rect.width / (num_points_for_graph - 1))
                        y_pt = graph_ui_rect.bottom - int(acc_hist * graph_ui_rect.height / 100)
                        points.append((x_pt, y_pt))
                    if len(points) > 1: pygame.draw.lines(self.screen, (255,0,0), False, points, 2)
                elif num_points_for_graph == 1:
                    x_pt = graph_ui_rect.x
                    y_pt = graph_ui_rect.bottom - int(self.accuracy_history[0] * graph_ui_rect.height / 100)
                    pygame.draw.circle(self.screen, (255,0,0), (x_pt, y_pt), 3)

        if self.editing_obstacles:
            # Enhanced instruction text for obstacle editing
            instr_lines = [
                "L-Click: Add new obstacle or drag existing",
                "R-Click: Delete obstacle", 
                "Shift+R-Click: Resize obstacle (grow)"
            ]
            for i, instruction in enumerate(instr_lines):
                instr_text = pygame.font.Font(None, 18).render(instruction, True, COLORS['text'])
                self.screen.blit(instr_text, (50, 660 + i * 15))
                
            # Visual feedback for dragging
            if self.dragging_obstacle is not None:
                ox, oy, radius = self.obstacles[self.dragging_obstacle]
                # Draw a dashed circle around the obstacle being dragged
                pygame.draw.circle(self.screen, (255, 255, 0), (int(ox), int(oy)), int(radius + 5), 3)
        
        pygame.display.flip()
    
    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(30)
        pygame.quit()

if __name__ == "__main__":
    game = RLSwarmGame()
    game.run() 