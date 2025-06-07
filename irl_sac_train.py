import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from env_rot import make_env

# Carica il dataset
data = np.load("trajectories/dataset_rot.npz")
observations = data["observations"]
actions = data["actions"]

episode_length = 100
n_episodes = len(observations) // episode_length

# Reward Network
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
        x = torch.cat([state, action], dim=-1)
        return self.model(x)

# Env wrapper con reward appreso
class IRLEnvWrapper(gym.Wrapper):
    def __init__(self, env, reward_net):
        super().__init__(env)
        self.reward_net = reward_net

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        with torch.no_grad():
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
            reward = self.reward_net(state_tensor, action_tensor).item()
        return obs, reward, terminated, truncated, info

# Funzione di training per la reward net
def train_reward_net(reward_net, expert_obs, expert_act, policy_obs, policy_act, optimizer, lambda_reg=1e-3):
    reward_net.train()
    expert_s = torch.tensor(expert_obs, dtype=torch.float32)
    expert_a = torch.tensor(expert_act, dtype=torch.float32)
    policy_s = torch.tensor(policy_obs, dtype=torch.float32)
    policy_a = torch.tensor(policy_act, dtype=torch.float32)

    r_expert = reward_net(expert_s, expert_a)
    r_policy = reward_net(policy_s, policy_a)

    loss = -r_expert.mean() + torch.logsumexp(r_policy, dim=0).mean()
    loss += lambda_reg * sum(torch.norm(p)**2 for p in reward_net.parameters())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# Inizializzazione
env = make_env()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
reward_net = RewardNetwork(state_dim, action_dim)
optimizer = optim.Adam(reward_net.parameters(), lr=1e-3)

# Inizializza SAC con reward fittizio
wrapped_env = DummyVecEnv([lambda: IRLEnvWrapper(make_env(), reward_net)])
agent = SAC("MlpPolicy", wrapped_env, verbose=1)

# Ciclo IRL
for iter in range(10):
    print(f"=== Iterazione IRL {iter} ===")
    agent.learn(total_timesteps=10000)

    # Raccogli dati policy
    policy_obs, policy_act = [], []
    obs = env.reset()
    for _ in range(1000):
        act, _ = agent.predict(obs, deterministic=True)
        new_obs, _, done, _, _ = env.step(act)
        policy_obs.append(obs)
        policy_act.append(act)
        obs = new_obs if not done else env.reset()

    policy_obs = np.array(policy_obs).squeeze()
    policy_act = np.array(policy_act).squeeze()

    # Prendi un batch casuale dagli esperti
    idx = np.random.choice(len(observations), size=policy_obs.shape[0], replace=False)
    expert_obs = observations[idx]
    expert_act = actions[idx]

    # Aggiorna reward
    loss = train_reward_net(reward_net, expert_obs, expert_act, policy_obs, policy_act, optimizer)
    print("Loss reward:", loss)

# Salva reward
torch.save(reward_net.state_dict(), "IL/DME_SAC/reward_network.pt")
