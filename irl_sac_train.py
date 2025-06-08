import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from env_rot import make_env

# Crea directory di salvataggio se non esiste
os.makedirs("IL/DME_SAC", exist_ok=True)

# Carica il dataset esperto
data = np.load("trajectories/dataset_rot.npz")
observations = data["observations"]
actions = data["actions"]

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
    return loss.item(), r_expert.mean().item(), r_policy.mean().item()

# Inizializzazione
env = make_env()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
reward_net = RewardNetwork(state_dim, action_dim)
optimizer = optim.Adam(reward_net.parameters(), lr=1e-3)

wrapped_env = DummyVecEnv([lambda: IRLEnvWrapper(make_env(), reward_net)])
agent = SAC("MlpPolicy", wrapped_env, verbose=1)

# Ciclo IRL
for iter in range(1000):
    print(f"=== Iterazione IRL {iter} ===")

    # 1. Raccogli dati della policy corrente
    policy_obs, policy_act = [], []
    obs, _ = env.reset()
    for _ in range(1000):
        act, _ = agent.predict(obs.reshape(1, -1), deterministic=True)
        new_obs, _, done, truncated, _ = env.step(act[0])
        policy_obs.append(obs)
        policy_act.append(act[0])
        
        obs = new_obs
        if truncated:  # Episodio finito
            obs, _ = env.reset()

    policy_obs = np.array(policy_obs).squeeze()
    policy_act = np.array(policy_act).squeeze()

    losses, r_exps, r_pols = [], [], []
    episode_len = 100
    n_policy_steps = len(policy_obs)
    n_episodes = n_policy_steps // episode_len
    n_expert_episodes = len(observations) // episode_len
    chosen_eps = np.random.choice(n_expert_episodes, size=n_episodes, replace=False)

    # 2. Allenamento multiplo della reward
    for _ in range(50):
        for i in chosen_eps:
            idx = i * episode_len
            expert_obs = observations[idx:idx+episode_len]
            expert_act = actions[idx:idx+episode_len]
            # Sottocampiona anche la policy per matchare l'esperto
            pol_idx = np.random.choice(len(policy_obs), size=episode_len, replace=False)
            policy_obs_batch = policy_obs[pol_idx]
            policy_act_batch = policy_act[pol_idx]
            loss, r_exp, r_pol= train_reward_net(reward_net, expert_obs, expert_act, policy_obs_batch, policy_act_batch, optimizer)
            losses.append(loss)
            r_exps.append(r_exp)
            r_pols.append(r_pol)

    #print("ultima loss", loss)
    #print("ultimo rew esperto", r_exp)
    #print("ultimo rew policy", r_pol)
    print(f"Loss reward (iter {iter}): {np.mean(losses):.2f}")
    print(f"Reward medio esperto: {np.mean(r_exps):.2f}")
    print(f"Reward medio policy: {np.mean(r_pols):.2f}")

    # 3. Aggiorna la policy ogni 5 iterazioni
    if iter % 5 == 0:
        print(">>> Aggiorno la policy con SAC")
        agent.learn(total_timesteps=2000)

# Salva il reward appreso
torch.save(reward_net.state_dict(), "IL/DME_SAC/reward_network_rot_0.5_0.01.pt")
print("Rete salvata")
