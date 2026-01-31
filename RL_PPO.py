import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
import torch
from stable_baselines3 import PPO 
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
        reward = 100 if current_pos in self.parking_slots else (-50 if current_pos in self.hurdles else -1)
        if current_pos in self.parking_slots or current_pos in self.hurdles:
            terminated = True
        return self.state, reward, terminated, False, {}

    def render(self, value_map=None, message=None):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("PPO Training & Heatmap")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.window.fill((240, 240, 240))

        # DRAW HEATMAP (If provided at the end)
        if value_map is not None:
            max_v, min_v = np.max(value_map), np.min(value_map)
            for y in range(self.size):
                for x in range(self.size):
                    val = value_map[y, x]
                    norm = (val - min_v) / (max_v - min_v + 1e-5)
                    # Green = High Value, Red = Low Value
                    color = (int(255 * (1 - norm)), int(255 * norm), 100)
                    pygame.draw.rect(self.window, color, (x*self.cell_size, y*self.cell_size, self.cell_size, self.cell_size))

        # Draw Static Elements
        for slot in self.parking_slots:
            pygame.draw.rect(self.window, (60, 60, 60), (slot[0]*self.cell_size, slot[1]*self.cell_size, self.cell_size, self.cell_size))
        for hurdle in self.hurdles:
            pygame.draw.rect(self.window, (200, 0, 0), (hurdle[0]*self.cell_size, hurdle[1]*self.cell_size, self.cell_size, self.cell_size))
        
        # Draw Agent
        pygame.draw.circle(self.window, (0, 100, 255), (int(self.state[0]*self.cell_size + self.cell_size/2), int(self.state[1]*self.cell_size + self.cell_size/2)), 15)

        if message:
            font = pygame.font.SysFont("Arial", 20, bold=True)
            text = font.render(message, True, (0, 0, 0))
            self.window.blit(text, (10, self.window_size - 30))

        pygame.display.flip()

# --- 2. THE CALLBACKS ---
class RenderCallback(BaseCallback):
    def __init__(self, env):
        super(RenderCallback, self).__init__()
        self.env = env
    def _on_step(self) -> bool:
        if pygame.display.get_init():
            pygame.event.pump()
        self.env.render()
        return True

# --- 3. HEATMAP EXTRACTION ---
def get_ppo_value_map(model, env_size):
    """Asks the PPO 'Critic' how much it likes each square"""
    v_map = np.zeros((env_size, env_size))
    for y in range(env_size):
        for x in range(env_size):
            obs = np.array([x, y], dtype=np.float32)
            with torch.no_grad():
                # Convert observation to tensor for the Neural Network
                obs_t = torch.as_tensor(obs).unsqueeze(0).to(model.device)
                value = model.policy.predict_values(obs_t)
                v_map[y, x] = value.item()
    return v_map

# --- 4. MAIN ---
if __name__ == "__main__":
    env = VisualParkingEnv()
    
    # EARLY STOPPING LOGIC
    # Stops training when Mean Reward reaches 95
    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=95, verbose=1)
    eval_cb = EvalCallback(env, callback_on_new_best=stop_cb, eval_freq=1000)
    
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=128)

    print("Training... Will stop early if car masters the grid.")
    try:
        model.learn(total_timesteps=30000, callback=CallbackList([RenderCallback(env), eval_cb]))
    except KeyboardInterrupt:
        pass

    # SAVE AND SHOW HEATMAP
    model.save("ppo_parking_final")
    print("Training Complete. Generating Heatmap...")
    
    final_heatmap = get_ppo_value_map(model, env.size)
    
    # Keep window open to see the heatmap
    while True:
        env.render(value_map=final_heatmap, message="FINAL HEATMAP: GREEN IS GOOD")