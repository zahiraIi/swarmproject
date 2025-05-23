"""
Interactive Visualization: Real-time swarm simulation with performance metrics
Beautiful pygame-based interface for demonstrating swarm behaviors
"""

import pygame
import numpy as np
import math
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
from swarm_engine import SwarmEnvironment, Agent
import sys
import time

# Color scheme
COLORS = {
    'background': (15, 25, 35),
    'agent': (100, 200, 255),
    'agent_highlight': (255, 255, 100),
    'obstacle': (255, 100, 100),
    'obstacle_glow': (255, 150, 150),
    'text': (255, 255, 255),
    'panel': (30, 45, 60),
    'button': (70, 130, 180),
    'button_hover': (100, 160, 210),
    'grid': (40, 60, 80),
    'velocity_line': (150, 255, 150)
}

class InteractiveSwarmVisualization:
    def __init__(self, width: int = 1000, height: int = 700):
        pygame.init()
        
        self.width = width
        self.height = height
        self.sim_width = 700
        self.sim_height = 600
        self.control_width = width - self.sim_width
        
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Swarm Robotics Simulation")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # simulation stuff
        self.swarm = SwarmEnvironment(self.sim_width, self.sim_height, 15)
        self.running = True
        self.paused = False
        
        # ui controls
        self.sliders = {
            'separation': {'value': 2.0, 'min': 0.5, 'max': 5.0, 'pos': (750, 100)},
            'alignment': {'value': 1.0, 'min': 0.0, 'max': 3.0, 'pos': (750, 150)}, 
            'cohesion': {'value': 1.0, 'min': 0.0, 'max': 3.0, 'pos': (750, 200)},
            'num_agents': {'value': 15, 'min': 5, 'max': 30, 'pos': (750, 250)}
        }
        
        self.buttons = {
            'pause': {'rect': pygame.Rect(750, 300, 100, 30), 'active': False},
            'reset': {'rect': pygame.Rect(870, 300, 100, 30), 'active': False},
            'clear_obstacles': {'rect': pygame.Rect(750, 340, 220, 30), 'active': False}
        }
        
        # tracking
        self.metrics_history = []
        self.max_history = 200
        
        # interaction
        self.dragging_slider = None
        
    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # left click
                    self._handle_left_click(event.pos)
                elif event.button == 3:  # right click
                    if event.pos[0] < self.sim_width:
                        self.swarm.add_obstacle(event.pos[0], event.pos[1], 
                                               np.random.uniform(20, 40))
                        
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging_slider = None
                    
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging_slider:
                    self._update_slider_value(event.pos)
                    
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self._reset_simulation()
                elif event.key == pygame.K_c:
                    self.swarm.remove_obstacles()
    
    def _handle_left_click(self, pos: Tuple[int, int]) -> None:
        # check buttons
        for name, button in self.buttons.items():
            if button['rect'].collidepoint(pos):
                if name == 'pause':
                    self.paused = not self.paused
                elif name == 'reset':
                    self._reset_simulation()
                elif name == 'clear_obstacles':
                    self.swarm.remove_obstacles()
                return
        
        # check sliders
        for name, slider in self.sliders.items():
            slider_rect = pygame.Rect(slider['pos'][0], slider['pos'][1] - 10, 200, 20)
            if slider_rect.collidepoint(pos):
                self.dragging_slider = name
                self._update_slider_value(pos)
                return
    
    def _update_slider_value(self, mouse_pos: Tuple[int, int]) -> None:
        if not self.dragging_slider:
            return
            
        slider = self.sliders[self.dragging_slider]
        slider_x = slider['pos'][0]
        
        # calculate new value
        relative_x = max(0, min(200, mouse_pos[0] - slider_x))
        ratio = relative_x / 200
        new_value = slider['min'] + ratio * (slider['max'] - slider['min'])
        
        slider['value'] = new_value
        
        # apply changes
        if self.dragging_slider == 'num_agents':
            target_count = int(new_value)
            current_count = len(self.swarm.agents)
            
            if target_count > current_count:
                for _ in range(target_count - current_count):
                    x = np.random.uniform(50, self.sim_width - 50)
                    y = np.random.uniform(50, self.sim_height - 50)
                    from swarm_engine import Agent
                    self.swarm.agents.append(Agent(x, y, self.sim_width, self.sim_height))
            elif target_count < current_count:
                self.swarm.agents = self.swarm.agents[:target_count]
        
        # update agent behavior weights
        for agent in self.swarm.agents:
            agent.separation_weight = self.sliders['separation']['value']
            agent.alignment_weight = self.sliders['alignment']['value']
            agent.cohesion_weight = self.sliders['cohesion']['value']
    
    def _reset_simulation(self) -> None:
        num_agents = int(self.sliders['num_agents']['value'])
        self.swarm = SwarmEnvironment(self.sim_width, self.sim_height, num_agents)
        self.metrics_history.clear()
        
    def update(self) -> None:
        if not self.paused:
            self.swarm.update()
            
            # store metrics
            if len(self.metrics_history) >= self.max_history:
                self.metrics_history.pop(0)
                
            self.metrics_history.append({
                'cohesion': self.swarm.swarm_cohesion,
                'speed': self.swarm.average_speed,
                'collisions': self.swarm.total_collisions
            })
    
    def draw(self) -> None:
        self.screen.fill(COLORS['background'])
        
        # draw simulation area
        pygame.draw.rect(self.screen, COLORS['grid'], (0, 0, self.sim_width, self.sim_height), 2)
        
        # grid background  
        for x in range(0, self.sim_width, 50):
            pygame.draw.line(self.screen, COLORS['grid'], (x, 0), (x, self.sim_height))
        for y in range(0, self.sim_height, 50):
            pygame.draw.line(self.screen, COLORS['grid'], (0, y), (self.sim_width, y))
        
        # draw obstacles
        for ox, oy, radius in self.swarm.obstacles:
            pygame.draw.circle(self.screen, COLORS['obstacle'], (int(ox), int(oy)), int(radius))
            pygame.draw.circle(self.screen, COLORS['obstacle_glow'], (int(ox), int(oy)), int(radius), 1)
        
        # draw agents
        for agent in self.swarm.agents:
            x, y = int(agent.position[0]), int(agent.position[1])
            pygame.draw.circle(self.screen, COLORS['agent_highlight'], (x, y), 8)
            
            # velocity vector
            vel_end = (
                x + agent.velocity[0] * 10,
                y + agent.velocity[1] * 10
            )
            pygame.draw.line(self.screen, COLORS['velocity_line'], (x, y), vel_end, 2)
        
        # ui panel
        self._draw_control_panel()
        self._draw_metrics()
        self._draw_performance_graph()
        
        pygame.display.flip()
    
    def _draw_control_panel(self) -> None:
        # panel background
        panel_rect = pygame.Rect(self.sim_width, 0, self.control_width, self.height)
        pygame.draw.rect(self.screen, COLORS['panel'], panel_rect)
        pygame.draw.line(self.screen, COLORS['grid'], (self.sim_width, 0), (self.sim_width, self.height), 2)
        
        # title
        title = self.font.render("SWARM CONTROL", True, COLORS['text'])
        self.screen.blit(title, (self.sim_width + 10, 10))
        
        # sliders
        y_offset = 50
        for name, slider in self.sliders.items():
            # label
            label = self.small_font.render(f"{name.title()}: {slider['value']:.1f}", True, COLORS['text'])
            self.screen.blit(label, (slider['pos'][0], slider['pos'][1] - 30))
            
            # slider track
            track_rect = pygame.Rect(slider['pos'][0], slider['pos'][1] - 5, 200, 10)
            pygame.draw.rect(self.screen, COLORS['grid'], track_rect)
            
            # slider handle
            ratio = (slider['value'] - slider['min']) / (slider['max'] - slider['min'])
            handle_x = slider['pos'][0] + ratio * 200
            pygame.draw.circle(self.screen, COLORS['button'], (int(handle_x), slider['pos'][1]), 8)
        
        # buttons
        for name, button in self.buttons.items():
            color = COLORS['button_hover'] if name == 'pause' and self.paused else COLORS['button']
            pygame.draw.rect(self.screen, color, button['rect'])
            pygame.draw.rect(self.screen, COLORS['text'], button['rect'], 2)
            
            text = self.small_font.render(name.title(), True, COLORS['text'])
            text_rect = text.get_rect(center=button['rect'].center)
            self.screen.blit(text, text_rect)
    
    def _draw_metrics(self) -> None:
        if not self.metrics_history:
            return
            
        y_start = 400
        metrics = self.metrics_history[-1]
        
        # current metrics
        metric_texts = [
            f"Agents: {len(self.swarm.agents)}",
            f"Obstacles: {len(self.swarm.obstacles)}",
            f"Cohesion: {metrics['cohesion']:.3f}",
            f"Avg Speed: {metrics['speed']:.2f}", 
            f"Collisions: {metrics['collisions']}"
        ]
        
        for i, text in enumerate(metric_texts):
            rendered = self.small_font.render(text, True, COLORS['text'])
            self.screen.blit(rendered, (self.sim_width + 10, y_start + i * 25))
    
    def _draw_performance_graph(self) -> None:
        if len(self.metrics_history) < 2:
            return
            
        graph_rect = pygame.Rect(self.sim_width + 10, 550, 280, 120)
        pygame.draw.rect(self.screen, COLORS['text'], graph_rect)
        pygame.draw.rect(self.screen, COLORS['grid'], graph_rect, 2)
        
        # graph data
        cohesion_values = [m['cohesion'] for m in self.metrics_history]
        
        if len(cohesion_values) > 1:
            max_val = max(cohesion_values) if max(cohesion_values) > 0 else 1
            
            # draw line
            points = []
            for i, val in enumerate(cohesion_values):
                x = graph_rect.left + (i / len(cohesion_values)) * graph_rect.width
                y = graph_rect.bottom - (val / max_val) * graph_rect.height
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, COLORS['velocity_line'], False, points, 2)
        
        # labels
        title = self.small_font.render("Cohesion Over Time", True, COLORS['text'])
        self.screen.blit(title, (graph_rect.left + 5, graph_rect.top + 5))
    
    def run(self) -> None:
        print("Starting interactive swarm simulation...")
        print("Controls:")
        print("  Left click: Interact with UI")
        print("  Right click: Add obstacles")
        print("  SPACE: Pause/unpause")
        print("  R: Reset simulation")
        print("  C: Clear obstacles")
        
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()

def main():
    try:
        viz = InteractiveSwarmVisualization()
        viz.run()
    except Exception as e:
        print(f"Error running visualization: {e}")
        print("Try running simple_demo.py instead")
        sys.exit(1)

if __name__ == "__main__":
    main() 