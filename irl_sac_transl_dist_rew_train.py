import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import torch.nn.functional as F

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
    def __init__(self, state_dim=2, action_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)

    def forward(self, state, action):
        #if action.ndim == 1:
        #    action = action.unsqueeze(1)
        x = torch.cat([state, action], dim=-1)
        return self.model(x)

# Env wrapper con reward appreso
class IRLEnvWrapper(gym.Wrapper):
    def __init__(self, env, reward_net):
        super().__init__(env)
        self.reward_net = reward_net

    def step(self, action):
        state = self.env.data.qpos
        #state = torch.tensor(state, dtype=torch.float32, device=self.reward_net.model[0].weight.device)
        #action = torch.tensor(action, dtype=torch.float32, device=self.reward_net.model[0].weight.device)
        obs, _, terminated, truncated, info = self.env.step(action)[:5]
        #step_result = self.env.step(action)
    
        #if len(step_result) == 6:
        #    obs, _, terminated, truncated, info, _ = step_result  # Scarta il sesto valore
        #else:
        #    obs, _, terminated, truncated, info = step_result
        with torch.no_grad():
            dist_x = state[2] - state[0]  # Calcola la distanza tra x e x target
            dist_y = state[3] - state[1]  # Calcola la distanza tra y e y target
            dist_tensor = torch.tensor([dist_x, dist_y], dtype=torch.float32, device=self.reward_net.model[0].weight.device).unsqueeze(0)
            #action_prep = preprocess_action(action)
            #state_tensor = torch.tensor(obs, dtype=torch.float32, device=self.reward_net.model[0].weight.device).unsqueeze(0)
            action_tensor = torch.tensor(action, dtype=torch.float32, device=self.reward_net.model[0].weight.device).unsqueeze(0)

            if action_tensor.ndim == 1:
                action_tensor = action_tensor.unsqueeze(1)

            reward = self.reward_net(dist_tensor, action_tensor).item()
        return obs, reward, terminated, truncated, info

# def preprocess_action(state, action):
#     # Caso batch
#     state = torch.tensor(state, dtype=torch.float32, device=device)
#     action = torch.tensor(action, dtype=torch.float32, device=device)
#     if state.ndim == 2 and action.ndim == 2:
#         pos = state[:, :2]      # (B, 2)
#         target = state[:, 2:4]  # (B, 2)
#         to_target = F.normalize(target - pos, dim=1)
#         action_dir = F.normalize(action, dim=1)
#         direction_reward = torch.sum(action_dir * to_target, dim=1)
#         direction_penalty = 1.0 - direction_reward
#         return direction_penalty.unsqueeze(1)  # (B, 1)

#     # Caso singolo
#     elif state.ndim == 1 and action.ndim == 1:
#         pos = state[:2]
#         target = state[2:4]
#         to_target = F.normalize(target - pos, dim=0)
#         action_dir = F.normalize(action, dim=0)
#         direction_reward = torch.dot(action_dir, to_target)
#         direction_penalty = 1.0 - direction_reward
# #         return direction_penalty.unsqueeze(0)  # (1,)

#     else:
#         raise ValueError("Stato e azione devono essere entrambi batch (2D) o entrambi singoli (1D).")

# def preprocess_action(action):
#     action = torch.tensor(action, dtype=torch.float32, device=device)
#     # Caso batch
#     if action.ndim == 2:
#         return F.normalize(action, dim=1)  # (B, 2)
#     elif action.ndim == 1:
#         return F.normalize(action, dim=0).unsqueeze(0)  # (1, 2)
#     else:
#         raise ValueError("Dimensioni non compatibili")

# Funzione di training per la reward net
def train_reward_net(reward_net, expert_obs, expert_act, policy_obs, policy_act, optimizer, lambda_reg=1e-3):
    reward_net.train()
    expert_s = torch.tensor(expert_obs, dtype=torch.float32, device=device)
    expert_a = torch.tensor(expert_act, dtype=torch.float32, device=device)
    policy_s = torch.tensor(policy_obs, dtype=torch.float32, device=device)
    policy_a = torch.tensor(policy_act, dtype=torch.float32, device=device)

    #expert_action_prep = preprocess_action(expert_a)  # (B, 1)
    #policy_action_prep = preprocess_action(policy_a)  # (B, 1)

    dist_expert = expert_s[:, 2:4] - expert_s[:, 0:2]  # Calcola la distanza tra x e x target
    dist_policy = policy_s[:, 2:4] - policy_s[:, 0:2]
    #expert_dist_tensor = torch.tensor(dist_expert, dtype=torch.float32, device=device)
    #policy_dist_tensor = torch.tensor(dist_policy, dtype=torch.float32, device=device)


    r_expert = reward_net(dist_expert, expert_a)
    r_policy = reward_net(dist_policy, policy_a)

    loss = -r_expert.mean() + torch.logsumexp(r_policy, dim=0).mean()
    loss += lambda_reg * sum(torch.norm(p)**2 for p in reward_net.parameters())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

if __name__ == "__main__":
    # Inizializzazione
    env = make_env()
    state_dim = 2 # relative distance between x and x target, y and y target
    action_dim = 2 # env.action_space.shape[0]
    reward_net = RewardNetwork(state_dim, action_dim).to(device)
    optimizer = optim.Adam(reward_net.parameters(), lr=1e-3)

    wrapped_env = DummyVecEnv([lambda: IRLEnvWrapper(make_env(), reward_net)])
    agent = SAC("MlpPolicy", wrapped_env, verbose=1, device=device)

    # Ciclo IRL
    for iter in range(1000):
        print(f"=== Iterazione IRL {iter} ===")

        # 1. Raccogli dati della policy corrente
        policy_obs, policy_act = [], []
        obs, _ = env.reset()
        for _ in range(1000):
            act, _ = agent.predict(obs.reshape(1, -1), deterministic=True)
            new_obs, _, done, truncated, _ = env.step(act[0])[:5]
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
    torch.save(reward_net.state_dict(), "IL/DME_SAC/reward_network_transl_0.2_0.05_dist_rew.pt")

    agent.save("IL/SAC_POLICY/sac_with_learned_reward_transl_0.2_0.05_dist_rew_IRL_")
    print("Modello SAC salvato.")