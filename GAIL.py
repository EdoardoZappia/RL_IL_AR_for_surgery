import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from rototransl_env import TrackingEnv

# === POLICY ===
class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim=6, act_dim=3):
        super().__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        mean = self.mean_net(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        return dist

# === DISCRIMINATORE ===
class Discriminator(nn.Module):
    def __init__(self, obs_dim=6, act_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.net(x)

# === UTILS ===
def rollout(policy, env, max_steps=100):
    obs_list, act_list = [], []
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)

    for _ in range(max_steps):
        with torch.no_grad():
            dist = policy(obs.unsqueeze(0))
            action = dist.sample().squeeze(0)
        obs_list.append(obs.numpy())
        act_list.append(action.numpy())
        next_obs, _, done, truncated, _, _ = env.step(action)
        if truncated:
            break
        obs = torch.tensor(next_obs, dtype=torch.float32)

    return np.array(obs_list), np.array(act_list)

# === TRAINING LOOP ===
def train_gail(policy, discriminator, expert_data, num_iterations=50000, device="cpu"):
    env = TrackingEnv()
    disc_optim = optim.Adam(discriminator.parameters(), lr=1e-3)
    policy_optim = optim.Adam(policy.parameters(), lr=1e-3)
    
    expert_obs = torch.tensor(expert_data["observations"], dtype=torch.float32)
    expert_acts = torch.tensor(expert_data["actions"], dtype=torch.float32)

    for it in range(num_iterations):
        # 1. Rollout policy
        agent_obs_np, agent_act_np = rollout(policy, env)
        agent_obs = torch.tensor(agent_obs_np, dtype=torch.float32)
        agent_acts = torch.tensor(agent_act_np, dtype=torch.float32)

        # 2. Train discriminator
        disc_optim.zero_grad()

        # --- Prepara batch dello stesso size ---
        N = agent_obs.shape[0]
        idx = torch.randint(0, expert_obs.shape[0], (N,))
        expert_batch_obs = expert_obs[idx]
        expert_batch_acts = expert_acts[idx]

        # --- Calcola logits ---
        expert_logits = discriminator(expert_batch_obs, expert_batch_acts)
        agent_logits = discriminator(agent_obs, agent_acts)

        # --- Loss ---
        loss_disc = -torch.mean(torch.log(expert_logits + 1e-8) + torch.log(1 - agent_logits + 1e-8))

        loss_disc.backward()
        disc_optim.step()

        # sample azione e log_prob
        dist = policy(agent_obs)
        sampled_action = dist.rsample()
        log_probs = dist.log_prob(sampled_action).sum(dim=-1)

        # reward da D
        with torch.no_grad():
            rewards = -torch.log(1 - discriminator(agent_obs, sampled_action) + 1e-8).squeeze()

        # policy loss
        loss_policy = -(log_probs * rewards).mean()

        # 4. Policy update (simple REINFORCE-style)
        policy_optim.zero_grad()
        loss_policy.backward()
        policy_optim.step()

        print(f"Iter {it} | Disc loss: {loss_disc.item():.4f} | Policy loss: {loss_policy.item():.4f} | Reward mean: {rewards.mean().item():.4f}")

    env.close()
    torch.save(policy.state_dict(), "IL/gail_policy.pth")
    print("\nGAIL training terminato e policy salvata in 'IL/gail_policy.pth'")

# === AVVIO ===
if __name__ == "__main__":
    expert_data = np.load("trajectories/dataset_filtered.npz")
    policy = GaussianPolicy()
    discriminator = Discriminator()
    train_gail(policy, discriminator, expert_data)
