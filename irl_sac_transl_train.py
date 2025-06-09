import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from environment import make_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Crea directory di salvataggio se non esiste
os.makedirs("IL/DME_SAC", exist_ok=True)
os.makedirs("IL/SAC_POLICY", exist_ok=True)

# Carica il dataset esperto
data = np.load("trajectories/dataset_transl.npz")
observations = data["observations"]
actions = data["actions"]

# Reward Network
class RewardNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)

    def forward(self, state, action):
        if action.ndim == 1:
            action = action.unsqueeze(1)
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
            state_tensor = torch.tensor(obs, dtype=torch.float32, device=self.reward_net.model[0].weight.device).unsqueeze(0)
            action_tensor = torch.tensor(action, dtype=torch.float32, device=self.reward_net.model[0].weight.device).unsqueeze(0)
            reward = self.reward_net(state_tensor, action_tensor).item()
        return obs, reward, terminated, truncated, info, _


# Funzione di training per la reward net
def train_reward_net(reward_net, expert_obs, expert_act, policy_obs, policy_act, optimizer, lambda_reg=1e-3):
    reward_net.train()
    expert_s = torch.tensor(expert_obs, dtype=torch.float32, device=device)
    expert_a = torch.tensor(expert_act, dtype=torch.float32, device=device)
    policy_s = torch.tensor(policy_obs, dtype=torch.float32, device=device)
    policy_a = torch.tensor(policy_act, dtype=torch.float32, device=device)

    if expert_a.ndim == 1:
        expert_a = expert_a.unsqueeze(1)
    if policy_a.ndim == 1:
        policy_a = policy_a.unsqueeze(1)

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
reward_net = RewardNetwork(state_dim, action_dim).to(device)
optimizer = optim.Adam(reward_net.parameters(), lr=1e-3)

wrapped_env = DummyVecEnv([lambda: IRLEnvWrapper(make_env(), reward_net)])
agent = SAC("MlpPolicy", wrapped_env, verbose=1, device=device)

# Ciclo IRL
for iter in range(1500):
    print(f"=== Iterazione IRL {iter} ===")

    # 1. Raccogli dati della policy corrente
    policy_obs, policy_act = [], []
    obs, _ = env.reset()
    for _ in range(1000):
        # act, _ = agent.predict(obs.reshape(1, -1), deterministic=True)
        # new_obs, _, done, _, _ = env.step(act[0])
        # policy_obs.append(obs)
        # policy_act.append(act[0])
        # obs = new_obs if not done else env.reset()[0]
        act, _ = agent.predict(obs.reshape(1, -1), deterministic=True)
        new_obs, _, done, truncated, _, _ = env.step(act[0])
        policy_obs.append(obs)
        policy_act.append(act[0])
        
        obs = new_obs
        if truncated:  # Episodio finito
            obs, _ = env.reset()

    policy_obs = np.array(policy_obs).squeeze()
    policy_act = np.array(policy_act).squeeze()

    # # 2. Allenamento multiplo della reward
    for _ in range(10):
        idx = np.random.choice(len(observations), size=policy_obs.shape[0], replace=False)
        expert_obs = observations[idx]
        expert_act = actions[idx]
        loss = train_reward_net(reward_net, expert_obs, expert_act, policy_obs, policy_act, optimizer)
    
    print(f"Loss reward (iter {iter}): {loss}")

    # 3. Aggiorna la policy ogni 5 iterazioni
    if iter % 5 == 0:
        print(">>> Aggiorno la policy con SAC")
        agent.learn(total_timesteps=2000)

# Salva il reward appreso
torch.save(reward_net.state_dict(), "IL/DME_SAC/reward_network_transl_0.2_0.05.pt")

agent.save("IL/SAC_POLICY/sac_with_learned_reward_transl_0.2_0.05_IRL_")
print("Modello SAC salvato.")