import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from environment import TrackingEnv
import datetime

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

MAX_MEMORY = 400   #4000
NUM_NEURONS = 256
LR_CRITIC = 0.0001
LR_ACTOR = 0.0005
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.1 #0.2
K_EPOCH = 10
ENTROPY_COEFF = 0.02    #0.01
CHECKPOINT_INTERVAL = 100
EARLY_STOPPING_EPISODES = 30

now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = f"runs/ppo_run_target_dyn{now}"
os.makedirs(RUN_DIR, exist_ok=True)

def compute_advantages(rewards, values, dones, gamma=GAMMA, lam=LAMBDA):
    advantages = []
    gae = 0
    values = torch.cat([values, torch.tensor([0], dtype=values.dtype, device=values.device)])  # Append a 0 for the last state
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return torch.tensor(advantages, dtype=torch.float)

def compute_returns(rewards, dones, gamma=0.99):
    returns = []
    G = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        G = reward + gamma * G * (1 - done)  # Reset if episode ended
        returns.insert(0, G)
    return returns

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()

        self.noise_std = 0.00   # No noise
        
        self.fc1 = nn.Linear(state_dim, NUM_NEURONS)
        self.fc2 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        
        # Output per la media (mu)
        self.mu_layer = nn.Linear(NUM_NEURONS, action_dim)
        
        # Output per la deviazione standard (log_sigma)
        self.log_sigma_layer = nn.Linear(NUM_NEURONS, action_dim)

    def add_noise_to_target(self, state):
        state = state.clone()
        if state.dim() == 1:
            #state[2:4] += torch.normal(0.0, self.noise_std, size=(2,), device=state.device)    # target only
            state += torch.normal(0.0, self.noise_std, size=(4,), device=state.device)  # whole state
        else:
            #state[:, 2:4] += torch.normal(0.0, self.noise_std, size=state[:, 2:4].shape, device=state.device)  # target only
            state += torch.normal(0.0, self.noise_std, size=state[:, :4].shape, device=state.device)    # whole state
        return state


    def forward(self, state, training=True):
        
        #if training:
            #state = self.add_noise_to_target(state)

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Media delle azioni
        mu = self.mu_layer(x)
        mu = 5.0*torch.tanh(mu)    # azioni in [-5, 5]
        
        # Deviazione standard (softplus per garantire positività)
        log_sigma = self.log_sigma_layer(x)
        sigma = F.softplus(log_sigma) + 1e-5 # 1e-5 per evitare log(0)
        #sigma = F.softplus(log_sigma) + exploration_term + 1e-5 # 1e-5 per evitare log(0)

        return mu, sigma

class ValueNet(nn.Module):
    def __init__(self, num_inputs):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(num_inputs, NUM_NEURONS)
        self.fc2 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.fc3 = nn.Linear(NUM_NEURONS, 1)
        self.noise_std = 0.01

    def add_noise_to_target(self, state):
        state = state.clone()
        if state.dim() == 1:
            #state[2:4] += torch.normal(0.0, self.noise_std, size=(2,), device=state.device)    # target only
            state += torch.normal(0.0, self.noise_std, size=(4,), device=state.device)  # whole state
        else:
            #state[:, 2:4] += torch.normal(0.0, self.noise_std, size=state[:, 2:4].shape, device=state.device)  # target only
            state += torch.normal(0.0, self.noise_std, size=(4,), device=state.device)  # whole state
        return state
        
    def forward(self, x, training=True):
        #if training:
        #    x = self.add_noise_to_target(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PPOAgent(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(PPOAgent, self).__init__()
        self.actor = PolicyNet(num_inputs, num_actions)
        self.critic = ValueNet(num_inputs)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.buffer = []
    
    def get_action(self, state, training=True):
        mu, sigma = self.actor.forward(state, training)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.rsample()  # Usa reparametrization trick per il backprop
        log_prob = dist.log_prob(action).sum(dim=-1)  # Somma log-prob per dimensione azione
        return action.detach(), log_prob

    def store_transition(self, transition):
        self.buffer.append(transition)

    def reward_function(self, state, action, next_state, tolerance, rimbalzato, attached_counter):
        #reward = - torch.norm(next_state[:2] - state[2:4]) -1
        pos = state[:2]
        target = state[2:4]              # target(t)
        next_pos = next_state[:2]        # agent(t+1)

        to_target = F.normalize(target - pos, dim=0)
        action_dir = F.normalize(action, dim=0)
        direction_reward = torch.dot(action_dir, to_target)
        direction_penalty = 1.0 - direction_reward

        reward = - direction_penalty 

        if torch.norm(next_state[:2] - state[2:4]) < tolerance:
            reward += 100 #+ 0.5 * attached_counter
        
        if rimbalzato:
            reward -= 5

        return reward - 1
        
    
    def update(self):
        torch.autograd.set_detect_anomaly(True)
        states, actions, rewards, dones, old_log_probs = zip(*self.buffer)
        old_log_probs = torch.stack(old_log_probs)
        states = torch.stack(states)
        actions = torch.stack(actions).float()
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        values = self.critic(states, training=True).squeeze()
        advantages = compute_advantages(rewards, values.detach(), dones)
        returns = compute_returns(rewards, dones)  # Compute Monte Carlo returns
        returns = torch.tensor(returns, dtype=torch.float32)
        
        for _ in range(K_EPOCH):
            means, stds = self.actor(states, training=True)
            dist = torch.distributions.Normal(means, stds)
            entropy = dist.entropy().sum(dim=-1).mean()  # media sull’intero batch
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEFF * entropy
            values_pred =self.critic(states, training=True).squeeze()
            value_loss = F.mse_loss(values_pred, returns)

            total_loss = policy_loss + value_loss

            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            #policy_loss.backward()
            #value_loss.backward()
            total_loss.backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()
        
        self.buffer.clear()

def save_checkpoint(agent, episode):
    path = os.path.join(RUN_DIR, f"checkpoint_ep{episode}.pth")
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict()
    }, path)

def plot_and_save(rewards, successes):
    plt.figure()
    plt.plot(rewards, label='Total Reward')
    plt.plot(np.convolve(successes, np.ones(10)/10, mode='valid'), label='Success Rate (10)')
    plt.legend()
    plt.xlabel('Episode')
    plt.title('PPO Training Progress')
    plt.savefig(os.path.join(RUN_DIR, 'training_plot.png'))
    plt.close()

def save_trajectory_plot(trajectory, target_trajectory, episode, tag="trajectory"):
    trajectory = np.array(trajectory)
    target_trajectory = np.array(target_trajectory)
    plt.figure(figsize=(5, 5))
    plt.plot(trajectory[:, 0], trajectory[:, 1], label="Agente", color='blue')
    plt.plot(target_trajectory[:, 0], target_trajectory[:, 1], label="Target", color='red')
    plt.scatter(*trajectory[0], color='green', label='Start agente', s=100)
    plt.scatter(*target_trajectory[0], color='yellow', label='Start target', s=100)
    plt.scatter(*target_trajectory[-1], color='red', label='End agente', s=100)
    plt.scatter(target_trajectory[-5:, 0], target_trajectory[-5:, 1], color='orange', label='Ultimi target', s=10)
    plt.scatter(trajectory[-5:, 0], trajectory[-5:, 1], color='purple', label='Ultimi agente', s=10)
    plt.title(f"{tag.capitalize()} - Episodio {episode}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.savefig(os.path.join(RUN_DIR, f"{tag}_ep{episode}.png"))
    plt.close()

def train_ppo(env=None, num_episodes=10001):
    if env is None:
        env = TrackingEnv()
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    agent = PPOAgent(num_inputs, num_actions)
    reward_history = []
    success_history = []
    counter = 0
    tolerance = 0.02

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        state = torch.tensor(state, dtype=torch.float32)
        trajectory = []
        target_trajectory = []
        step = 0
        attached_counter = 0
        total_attached_counter = 0

        while not done:
            step += 1
            trajectory.append(state[:2].detach().numpy())
            target_trajectory.append(state[2:4].detach().numpy())
            action, log_prob = agent.get_action(state)
            next_state, _, done, truncated, _, rimbalzato = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)

            if torch.norm(next_state[:2] - state[2:4]) > tolerance:
                attached_counter = 0
            else:
                attached_counter += 1
                total_attached_counter += 1

            if attached_counter > 20 or truncated or (total_attached_counter > 0 and attached_counter == 0):
                done = True

            reward = agent.reward_function(state, action, next_state, tolerance, rimbalzato, attached_counter)
            agent.store_transition((state, action, reward, done, log_prob))
            state = next_state
            total_reward += reward

        if attached_counter > 20:
            counter += 1
            success_history.append(1)
            if counter % 100 == 0:
                save_trajectory_plot(trajectory, target_trajectory, episode, tag="success")
        else:
            success_history.append(0)

        reward_history.append(total_reward)

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {total_reward:.2f}, Attached_counter: {attached_counter}, Total attached counter: {total_attached_counter}, Successes: {counter}")

        if episode % 100 == 0:
            #print(f"Episode: {episode}, Reward: {total_reward:.2f}, Successes: {counter}")
            save_trajectory_plot(trajectory, target_trajectory, episode)
            save_checkpoint(agent, episode)

        if len(reward_history) > EARLY_STOPPING_EPISODES and np.mean(reward_history[-EARLY_STOPPING_EPISODES:]) > 2000:
           print(f"Early stopping at episode {episode}")
           save_checkpoint(agent, episode)
           save_trajectory_plot(trajectory, target_trajectory, episode)
           break

        if len(agent.buffer) >= MAX_MEMORY:
            agent.update()

    env.close()
    np.save(os.path.join(RUN_DIR, 'rewards.npy'), reward_history)
    np.save(os.path.join(RUN_DIR, 'successes.npy'), success_history)
    plot_and_save(reward_history, success_history)
    print("Training finished. Artifacts saved in:", RUN_DIR)
    return agent

if __name__ == "__main__":
    trained_agent = train_ppo()
