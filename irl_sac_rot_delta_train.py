import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from env_rot import make_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Crea directory di salvataggio se non esiste
os.makedirs("IL/DME_SAC", exist_ok=True)
os.makedirs("IL/SAC_POLICY", exist_ok=True)

# Carica il dataset esperto
data = np.load("trajectories/dataset_rot.npz")
observations = data["observations"]
actions = data["actions"]

# Reward Network
class RewardNetwork(nn.Module):
    def __init__(self, state_dim=2, action_dim=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)

    def forward(self, delta_before, action, delta_after):
        x = torch.cat([delta_before, action, delta_after], dim=-1)
        return self.model(x)

# Env wrapper con reward appreso
class IRLEnvWrapper(gym.Wrapper):
    def __init__(self, env, reward_net):
        super().__init__(env)
        self.reward_net = reward_net

    def step(self, action):
        last_obs = self.env.data.qpos
        theta_t = last_obs[0]
        target_t = last_obs[1]

        obs, _, terminated, truncated, info = self.env.step(action)
        theta_tp1 = obs[0]  # nuovo theta

        delta_before = target_t - theta_t
        delta_after = target_t - theta_tp1  # target_t resta fisso

        with torch.no_grad():
            delta_before_tensor = torch.tensor(delta_before, dtype=torch.float32, device=device).view(1, 1)
            delta_after_tensor  = torch.tensor(delta_after,  dtype=torch.float32, device=device).view(1, 1)

            if isinstance(action, np.ndarray):
                action_tensor = torch.tensor(action, dtype=torch.float32, device=device).view(1, -1)
            else:
                action_tensor = torch.tensor([action], dtype=torch.float32, device=device).view(1, 1)

            reward = self.reward_net(delta_before_tensor, action_tensor, delta_after_tensor).item()

        self.last_obs = obs  # aggiorna lo stato corrente
        return obs, reward, terminated, truncated, info


def train_reward_net(reward_net, expert_obs, expert_act, policy_obs, policy_act, optimizer, lambda_reg=1e-3):
    reward_net.train()
    expert_a = torch.tensor(expert_act, dtype=torch.float32, device=device).unsqueeze(1)
    policy_a = torch.tensor(policy_act, dtype=torch.float32, device=device).unsqueeze(1)

    # Delta prima e dopo per dati esperti
    expert_delta_before = expert_obs[:-1, 1] - expert_obs[:-1, 0]
    expert_delta_after  = expert_obs[:-1, 1] - expert_obs[1:, 0]
    expert_delta_before = torch.tensor(expert_delta_before, dtype=torch.float32, device=device).unsqueeze(1)
    expert_delta_after  = torch.tensor(expert_delta_after,  dtype=torch.float32, device=device).unsqueeze(1)
    expert_a = expert_a[:-1]
    if expert_a.ndim == 1:
        expert_a = expert_a.unsqueeze(1)
    elif expert_a.ndim == 3:
        expert_a = expert_a.squeeze(2)

    # Delta prima e dopo per dati della policy
    policy_delta_before = policy_obs[:-1, 1] - policy_obs[:-1, 0]
    policy_delta_after  = policy_obs[:-1, 1] - policy_obs[1:, 0]
    policy_delta_before = torch.tensor(policy_delta_before, dtype=torch.float32, device=device).unsqueeze(1)
    policy_delta_after  = torch.tensor(policy_delta_after,  dtype=torch.float32, device=device).unsqueeze(1)
    policy_a = policy_a[:-1]
    if policy_a.ndim == 1:
        policy_a = policy_a.unsqueeze(1)
    elif policy_a.ndim == 3:
        policy_a = policy_a.squeeze(2)

    # Calcolo dei reward
    r_expert = reward_net(expert_delta_before, expert_a, expert_delta_after)
    r_policy = reward_net(policy_delta_before, policy_a, policy_delta_after)

    loss = -r_expert.mean() + torch.logsumexp(r_policy, dim=0).mean()
    loss += lambda_reg * sum(torch.norm(p)**2 for p in reward_net.parameters())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


if __name__ == "__main__":

    # Inizializzazione
    env = make_env()
    state_dim = 2 # relative distance between theta and theta target
    action_dim = 1 # rotation speed
    reward_net = RewardNetwork(state_dim, action_dim).to(device)
    optimizer = optim.Adam(reward_net.parameters(), lr=1e-3)

    wrapped_env = DummyVecEnv([lambda: IRLEnvWrapper(make_env(), reward_net)])
    agent = SAC("MlpPolicy", wrapped_env, verbose=1, device=device)

    # Ciclo IRL
    for iter in range(500):
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
            new_obs, _, done, truncated, _ = env.step(act[0])
            policy_obs.append(obs)
            policy_act.append(act[0])
            
            obs = new_obs
            if truncated:  # Episodio finito
                obs, _ = env.reset()

        policy_obs = np.array(policy_obs).squeeze()
        policy_act = np.array(policy_act).squeeze()

        # # 2. Allenamento multiplo della reward
        for _ in range(5):
            idx = np.random.choice(len(observations), size=policy_obs.shape[0], replace=False)
            expert_obs = observations[idx]
            expert_act = actions[idx]
            loss = train_reward_net(reward_net, expert_obs, expert_act, policy_obs, policy_act, optimizer)
        
        print(f"Loss reward (iter {iter}): {loss}")

        # 3. Aggiorna la policy ogni 5 iterazioni
        if iter % 2 == 0:
            print(">>> Aggiorno la policy con SAC")
            agent.learn(total_timesteps=1200)

    # Salva il reward appreso
    torch.save(reward_net.state_dict(), "IL/DME_SAC/reward_network_rot_0.5_0.01_delta.pt")

    agent.save("IL/SAC_POLICY/sac_with_learned_reward_rot_0.5_0.01_delta_IRL")
    print("Modello SAC salvato.")