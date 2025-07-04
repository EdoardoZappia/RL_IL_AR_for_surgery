import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from rototransl_env import TrackingEnv
import random
from collections import deque
import datetime

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

NUM_NEURONS = 256
LR_ACTOR = 0.001
LR_CRITIC = 0.001
GAMMA = 0.99
TAU = 0.005
EARLY_STOPPING_EPISODES = 30
CHECKPOINT_INTERVAL = 100

now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = f"runs_rototransl_static_2d/ddpg_run_1net_static{now}"
os.makedirs(RUN_DIR, exist_ok=True)

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, NUM_NEURONS)
        self.fc2 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.fc3 = nn.Linear(NUM_NEURONS, action_dim)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = self.fc3(x)
        action_xy = torch.tanh(action[:, :2]) * 5.0
        action_rot = torch.tanh(action[:, 2:3]) * 5.0
        return torch.cat([action_xy, action_rot], dim=1).squeeze(0)

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, NUM_NEURONS)
        self.fc2 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.fc3 = nn.Linear(NUM_NEURONS, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.relu(self.fc3(x))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DDPGAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DDPGAgent, self).__init__()
        self.actor = PolicyNet(state_dim, action_dim)
        self.actor_target = PolicyNet(state_dim, action_dim)
        self.critic = QNet(state_dim, action_dim)
        self.critic_target = QNet(state_dim, action_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.buffer = ReplayBuffer(50000) #statico e dinamico rumoroso
        #self.buffer = ReplayBuffer(20000) #dinamico
        self.batch_size = 128
        self.noise_std = 0.5
        self.min_noise_std = 0.01
        self.noise_decay = 0.999

    def reward_function(self, state, action, next_state, tolerance_transl, tolerance_rot, rimbalzato):
        pos = state[:2]
        target = state[3:5]              # target(t)
        next_pos = next_state[:2]        # agent(t+1)

        to_target = F.normalize(target - pos, dim=0)
        action_dir = F.normalize(action[:2], dim=0)
        direction_reward = torch.dot(action_dir, to_target)
        direction_penalty = 1.0 - direction_reward

        rot_error = torch.abs(next_state[2] - state[5])

        reward = - 5 * direction_penalty - 3 * rot_error

        #reward = - 3 * rot_error

        if torch.norm(next_state[:2] - state[3:5]) < tolerance_transl and torch.norm(next_state[2] - state[5]) < tolerance_rot:
            reward += 100
        
        #if torch.norm(next_state[:2] - state[3:5]) < tolerance_transl:
        #    reward += 5

        #if torch.norm(next_state[2] - state[5]) < tolerance_rot:
        #    reward += 100   #5

        #if rimbalzato:
        #    reward -= 100
        
        return reward - 1

    def update(self, gamma=GAMMA, tau=TAU, device='cpu'):
        if len(self.buffer) < self.batch_size:
            return
        transitions = random.sample(self.buffer.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, next_actions)
            y = rewards + gamma * target_Q * (1 - dones)

        current_Q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q, y)

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

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
    plt.title('DDPG Training Progress')
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

def train_ddpg(env=None, num_episodes=10001):
    if env is None:
        env = TrackingEnv()
    state_dim = env.observation_space.shape[0]
    print(state_dim)
    action_dim = env.action_space.shape[0]
    print(action_dim)
    agent = DDPGAgent(state_dim, action_dim)
    reward_history, success_history = [], []
    counter = 0
    tolerance_transl = 0.02
    tolerance_rot = 0.01

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        real_state = torch.tensor(state, dtype=torch.float32)
        state = torch.tensor(state, dtype=torch.float32)

        #state = state.clone()
        #state[2:4] += torch.normal(mean=0.0, std=0.005, size=(2,), device=state.device)

        agent.noise_std = max(agent.min_noise_std, agent.noise_std * agent.noise_decay)     # Exploration
        trajectory, target_trajectory = [], []
        attached_counter = 0
        total_attached_counter = 0

        while not done:
            trajectory.append(state[:2].detach().numpy())
            target_trajectory.append(state[3:5].detach().numpy())
            action = agent.actor(state).detach().numpy()
            noise = np.random.normal(0, agent.noise_std, size=action.shape)
            noisy_action = action + noise   # Exploration
            noisy_action = np.clip(noisy_action, env.action_space.low, env.action_space.high)
            action_tensor = torch.tensor(noisy_action, dtype=torch.float32)

            next_state, _, done, truncated, _, rimbalzato = env.step(noisy_action)

            real_next_state = torch.tensor(next_state, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            

            #next_state = next_state.clone()
            #next_state[2:4] += torch.normal(mean=0.0, std=0.005, size=(2,), device=next_state.device)

            # if torch.norm(real_next_state[:2] - real_state[3:5]) < tolerance_transl:
            #     print("RAGGIUNTO TARGET, episodio:", episode)

            # if torch.norm(real_next_state[2] - real_state[5]) < tolerance_rot:
            #     print("RAGGIUNTO TARGET ROTAZIONE, episodio:", episode)

            #if torch.norm(real_next_state[:2] - real_state[3:5]) < tolerance_transl and torch.norm(real_next_state[2] - real_state[5]) < tolerance_rot:
            if torch.norm(real_next_state[2] - real_state[5]) < tolerance_rot:
                total_attached_counter += 1
                attached_counter += 1
            else:
                attached_counter = 0

            reward = agent.reward_function(real_state, action_tensor, real_next_state, tolerance_transl, tolerance_rot, rimbalzato)

            if attached_counter == 1 or truncated:
                done = True

            #condition = torch.norm(real_next_state[:2] - real_state[3:5]) > tolerance_transl or torch.norm(real_next_state[2] - real_state[5]) > tolerance_rot
            #if attached_counter > 20 or truncated or (total_attached_counter > 0 and condition):
            #    done = True
            
            transition = (state.numpy(), action_tensor.numpy(), reward, next_state.numpy(), float(done))
            agent.buffer.push(transition)
            if len(agent.buffer) > 1000:
                agent.update()
            state = next_state
            real_state = real_next_state
            total_reward += reward


        if attached_counter == 1: #> 20:
            counter += 1
            success_history.append(1)
            if counter % 100 == 0:
                save_trajectory_plot(trajectory, target_trajectory, episode, tag="success")
        else:
            success_history.append(0)

        reward_history.append(total_reward)

        if total_attached_counter > 0:
            print(f"Episode {episode}, Reward: {total_reward:.2f}, Attached_counter: {attached_counter}, Total attached counter: {total_attached_counter}, Successes: {counter}")

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {total_reward:.2f}, Attached_counter: {attached_counter}, Total attached counter: {total_attached_counter}, Successes: {counter}")
        if episode % CHECKPOINT_INTERVAL == 0 and episode > 0:
            save_checkpoint(agent, episode)
        if episode % 50 == 0 and episode > 0:
            save_trajectory_plot(trajectory, target_trajectory, episode)

        if len(reward_history) > EARLY_STOPPING_EPISODES and np.mean(success_history[-EARLY_STOPPING_EPISODES:]) == 1:#np.mean(reward_history[-EARLY_STOPPING_EPISODES:]) > 2000:
           print(f"Early stopping at episode {episode}")
           save_checkpoint(agent, episode)
           save_trajectory_plot(trajectory, target_trajectory, episode)
           break

    np.save(os.path.join(RUN_DIR, 'rewards.npy'), reward_history)
    np.save(os.path.join(RUN_DIR, 'successes.npy'), success_history)
    plot_and_save(reward_history, success_history)
    env.close()
    return agent

if __name__ == "__main__":
    trained_agent = train_ddpg()
