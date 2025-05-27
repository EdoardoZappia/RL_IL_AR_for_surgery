import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from rototransl_env import TrackingEnv

# === POLICY ===
class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim=4, act_dim=2):
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

# === VALUE FUNCTION ===
class ValueFunction(nn.Module):
    def __init__(self, obs_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)

# === DISCRIMINATORE ===
class Discriminator(nn.Module):
    def __init__(self, obs_dim=4, act_dim=2):
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
    obs_list, act_list, log_probs = [], [], []
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)

    for _ in range(max_steps):
        obs_transl = torch.cat([state[:2], state[3:5]])
        dist = policy(obs_transl.unsqueeze(0))
        action = dist.sample().squeeze(0)
        log_prob = dist.log_prob(action).sum()

        # Costruisci azione completa per l'env
        full_action = torch.zeros(3)
        full_action[:2] = action  # dx, dy

        next_state, _, done, truncated, _, _ = env.step(full_action.detach().numpy())

        obs_list.append(obs_transl)
        act_list.append(action)
        log_probs.append(log_prob)

        state = torch.tensor(next_state, dtype=torch.float32)
        if truncated:
            break

    return torch.stack(obs_list), torch.stack(act_list), torch.stack(log_probs)

# === TRAINING LOOP ===
def train_gail(policy, discriminator, expert_data, num_iterations=1000, device="cpu"):
    env = TrackingEnv()
    value_fn = ValueFunction().to(device)

    disc_optim = optim.Adam(discriminator.parameters(), lr=1e-3)
    policy_optim = optim.Adam(policy.parameters(), lr=3e-4)
    value_optim = optim.Adam(value_fn.parameters(), lr=1e-3)

    expert_obs = torch.tensor(expert_data["observations"], dtype=torch.float32)
    expert_acts = torch.tensor(expert_data["actions"], dtype=torch.float32)

    for it in range(num_iterations):
        # 1. Rollout policy
        agent_obs, agent_acts, log_probs_old = rollout(policy, env)
        agent_obs = agent_obs.detach()
        agent_acts = agent_acts.detach()
        log_probs_old = log_probs_old.detach()

        # 2. Train discriminator
        disc_optim.zero_grad()
        N = agent_obs.shape[0]
        idx = torch.randint(0, expert_obs.shape[0], (N,))
        expert_batch_obs = expert_obs[idx]
        expert_batch_acts = expert_acts[idx]

        expert_logits = discriminator(expert_batch_obs, expert_batch_acts)
        agent_logits = discriminator(agent_obs, agent_acts)

        loss_disc = -torch.mean(torch.log(expert_logits + 1e-8) + torch.log(1 - agent_logits + 1e-8))
        loss_disc.backward()
        disc_optim.step()

        # 3. Compute rewards from discriminator
        with torch.no_grad():
            rewards = -torch.log(1 - discriminator(agent_obs, agent_acts) + 1e-8).squeeze()

        # 4. Compute advantage
        values = value_fn(agent_obs)
        returns = rewards
        advantages = returns - values.detach()

        # 5. PPO policy update
        dist = policy(agent_obs)
        log_probs = dist.log_prob(agent_acts).sum(dim=-1)
        ratio = torch.exp(log_probs - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        policy_optim.zero_grad()
        policy_loss.backward()
        policy_optim.step()

        # 6. Value function update
        value_loss = nn.MSELoss()(values, returns)
        value_optim.zero_grad()
        value_loss.backward()
        value_optim.step()

        print(f"Iter {it} | Disc loss: {loss_disc.item():.4f} | Policy loss: {policy_loss.item():.4f} | Value loss: {value_loss.item():.4f} | Reward mean: {rewards.mean().item():.4f}")

    env.close()
    torch.save(policy.state_dict(), "IL/gail_policy_transl.pth")
    print("\nGAIL training terminato e policy salvata in 'IL/gail_policy_transl.pth'")

# === AVVIO ===
if __name__ == "__main__":
    expert_data = np.load("trajectories/dataset_transl.npz")
    policy = GaussianPolicy()
    discriminator = Discriminator()
    train_gail(policy, discriminator, expert_data)
