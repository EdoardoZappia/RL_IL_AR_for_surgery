import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import os

from env_rot import make_env  # Assicurati che make_env sia definito

# Usa GPU se disponibile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica la rete di reward appresa
class RewardNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        if action.ndim == 1:
            action = action.unsqueeze(1)
        x = torch.cat([state, action], dim=-1)
        return self.model(x)

# Wrapper per usare la reward appresa
class IRLEnvWrapper(gym.Wrapper):
    def __init__(self, env, reward_net):
        super().__init__(env)
        self.reward_net = reward_net

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        with torch.no_grad():
            s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            reward = self.reward_net(s, a).item()
        return obs, reward, terminated, truncated, info

def train_sac_with_learned_reward():
    # Inizializza ambiente e reward network
    env = make_env()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    reward_net = RewardNetwork(state_dim, action_dim).to(device)
    reward_net.load_state_dict(torch.load("IL/DME_SAC/reward_network_rot_0.5_0.01.pt", map_location=device))
    reward_net.eval()

    # Crea ambiente con reward appresa
    wrapped_env = DummyVecEnv([lambda: IRLEnvWrapper(make_env(), reward_net)])

    # SAC con reward appresa
    model = SAC("MlpPolicy", wrapped_env, verbose=1, device=device)
    #model.set_parameters("IL/SAC_POLICY/sac_with_learned_reward_rot_0.5_0.01_IRL")

    # Allenamento
    model.learn(total_timesteps=300_000)

    # Crea la directory di salvataggio
    os.makedirs("IL/SAC_POLICY", exist_ok=True)

    # Salva l'intero modello
    model.save("IL/SAC_POLICY/sac_with_learned_reward_rot_0.5_0.01")

    print("Policy addestrata e salvata.")

if __name__ == "__main__":
    train_sac_with_learned_reward()
