import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback, CallbackList

# --- 1. THE ENVIRONMENT ---
class VisualParkingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    def __init__(self):
        super(VisualParkingEnv, self).__init__()
        self.size = 10
        self.cell_size = 50
        self.window_size = self.size * self.cell_size
        self.parking_slots = [(1,2), (1,5), (1,7), (5,2), (5,5), (5,7), (9,2), (9,5), (9,7)]
        self.hurdles = [(3,3), (3,4), (3,5), (7,3), (7,4), (7,5)]
        self.observation_space = spaces.Box(low=0, high=self.size-1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.window = None
        self.clock = None
        self.state = np.array([0, 0], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0, 0], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        if action == 0: self.state[1] = max(0, self.state[1] - 1)
        elif action == 1: self.state[1] = min(self.size-1, self.state[1] + 1)
        elif action == 2: self.state[0] = max(0, self.state[0] - 1)
        elif action == 3: self.state[0] = min(self.size-1, self.state[0] + 1)
        terminated = False
        current_pos = tuple(self.state.astype(int))
        if current_pos in self.parking_slots:
            reward = 100
            terminated = True
        elif current_pos in self.hurdles:
            reward = -50
            terminated = True
        else:
            reward = -1 
        return self.state, reward, terminated, False, {}

    def render(self, message=None, q_values=None):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("DQN Car Parking - Visual Q-Table")
            self.clock = pygame.time.Clock()

        self.window.fill((240, 240, 240))

        # --- NEW: DRAW Q-VALUE HEATMAP ---
        if q_values is not None:
            max_v = np.max(q_values)
            min_v = np.min(q_values)
            for y in range(self.size):
                for x in range(self.size):
                    val = q_values[y, x]
                    # Normalize color between Red (Low) and Green (High)
                    norm = (val - min_v) / (max_v - min_v + 1e-5)
                    color = (int(255 * (1 - norm)), int(255 * norm), 100)
                    pygame.draw.rect(self.window, color, (x*self.cell_size, y*self.cell_size, self.cell_size, self.cell_size))

        # Draw Grid, Slots, and Hurdles
        for x in range(0, self.window_size, self.cell_size):
            pygame.draw.line(self.window, (200, 200, 200), (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, self.cell_size):
            pygame.draw.line(self.window, (200, 200, 200), (0, y), (self.window_size, y))
        for slot in self.parking_slots:
            pygame.draw.rect(self.window, (60, 60, 60), (slot[0]*self.cell_size, slot[1]*self.cell_size, self.cell_size, self.cell_size))
        for hurdle in self.hurdles:
            pygame.draw.rect(self.window, (200, 0, 0), (hurdle[0]*self.cell_size, hurdle[1]*self.cell_size, self.cell_size, self.cell_size))
        
        # Draw Agent
        pygame.draw.circle(self.window, (0, 100, 255), (int(self.state[0]*self.cell_size + self.cell_size/2), int(self.state[1]*self.cell_size + self.cell_size/2)), 15)

        if message:
            font = pygame.font.SysFont("Arial", 30, bold=True)
            text = font.render(message, True, (255, 255, 255))
            text_rect = text.get_rect(center=(self.window_size//2, self.window_size//2))
            pygame.draw.rect(self.window, (0, 0, 0), text_rect.inflate(20, 20))
            self.window.blit(text, text_rect)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

# --- 2. THE VISUAL CALLBACKS ---
class RenderCallback(BaseCallback):
    def __init__(self, env):
        super(RenderCallback, self).__init__()
        self.env = env
    def _on_step(self) -> bool:
        self.env.render()
        return True

# --- 3. LOGIC TO EXTRACT HEATMAP DATA ---
def get_q_heatmap(model, env_size):
    q_map = np.zeros((env_size, env_size))
    for y in range(env_size):
        for x in range(env_size):
            obs = np.array([x, y], dtype=np.float32)
            with torch.no_grad():
                obs_t = torch.as_tensor(obs).unsqueeze(0).to(model.device)
                q_values = model.q_net(obs_t)
                q_map[y, x] = torch.max(q_values).item()
    return q_map

# --- 4. MAIN ---
if __name__ == "__main__":
    env = VisualParkingEnv()
    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=92, verbose=1)
    eval_cb = EvalCallback(env, callback_on_new_best=stop_cb, eval_freq=1000)
    render_cb = RenderCallback(env)
    
    model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-3)

    print("Training and generating Heatmap...")
    try:
        model.learn(total_timesteps=20000, callback=CallbackList([render_cb, eval_cb]))
    except KeyboardInterrupt:
        pass

    # SHOW FINAL VISUAL Q-TABLE
    final_q_map = get_q_heatmap(model, env.size)
    print("Showing Heatmap. Green = High confidence, Red = Danger.")
    
    running = True
    while running:
        env.render(message="Q-TABLE HEATMAP ACTIVE", q_values=final_q_map)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    
    pygame.quit()