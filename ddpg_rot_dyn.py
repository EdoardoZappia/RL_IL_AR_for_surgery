import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from env_rot import TrackingEnv
import random
from collections import deque
import datetime

# SEED = 42
# torch.manual_seed(SEED)
# np.random.seed(SEED)

NUM_NEURONS = 128 #256
LR_ACTOR = 0.001
LR_CRITIC = 0.001
GAMMA = 0.99
TAU = 0.005
EARLY_STOPPING_EPISODES = 50
CHECKPOINT_INTERVAL = 100

now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = f"Rotazioni-dinamiche/Noisy/ddpg_mov_0.01_std_0.003_{now}"
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
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) * 5.0
        return action

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
        return self.fc3(x)

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
        self.buffer = ReplayBuffer(50000) 
        #self.buffer = ReplayBuffer(20000)
        self.batch_size = 128
        self.noise_std = 0.5
        self.min_noise_std = 0.01
        self.noise_decay = 0.999

    def reward_function(self, state, action, next_state, tolerance):

        # theta = state[0]
        # theta_target = state[1]              # target(t)
        # next_theta = next_state[0]        # agent(t+1)

        # delta_before = theta_target - theta

        # # Direzione dell’azione
        # desired_sign = torch.sign(delta_before)
        # action_sign = torch.sign(action[0])
       
        rot_error = torch.norm(state[1]-next_state[0])

        # reward = 1.0 if action_sign == desired_sign else -1.0

        reward = - rot_error.item() * 3

        if torch.norm(next_state[0] - state[1]) < tolerance:
            reward += 100
        
        return reward - 1.0

    # def reward_function(self, state, action, next_state, tolerance=0.02):
    #     theta = state[0]           # orientamento agente attuale
    #     theta_target = state[1]    # orientamento target
    #     next_theta = next_state[0] # orientamento agente al passo successivo

    #     # Errore angolare prima e dopo l’azione
    #     delta_before = theta_target - theta
    #     delta_after = theta_target - next_theta

    #     # Miglioramento (quanto si è avvicinato)
    #     improvement = abs(delta_before) - abs(delta_after)

    #     # Direzione dell’azione
    #     desired_sign = torch.sign(delta_before)
    #     action_sign = torch.sign(action[0])

    #     direction_alignment = 1.0 if action_sign == desired_sign else -1.0

    #     # Reward combinato: miglioramento * allineamento direzionale
    #     reward = improvement * direction_alignment

    #     # Bonus se dentro la tolleranza
    #     if abs(delta_after) < tolerance:
    #         reward += 100

    #     return reward - 1  # penalità di base per incoraggiare rapidità


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
    plt.legend()
    plt.xlabel('Episode')
    plt.title('DDPG Training Progress')
    plt.savefig(os.path.join(RUN_DIR, 'training_plot.png'))
    plt.close()

def save_trajectory_plot(trajectory, target_trajectory, episode, tag="trajectory"):
    trajectory = np.array(trajectory)
    target_trajectory = np.array(target_trajectory)
    plt.figure(figsize=(6, 4))
    plt.plot(trajectory, label="Agente", color='blue')
    plt.plot(target_trajectory, label="Target", color='red')
    plt.scatter(0, trajectory[0], color='green', label='Start agente', s=60)
    plt.scatter(0, target_trajectory[0], color='yellow', label='Start target', s=60)
    plt.scatter(len(target_trajectory) - 1, target_trajectory[-1], color='red', label='End target', s=60)
    plt.scatter(len(trajectory) - 1, trajectory[-1], color='purple', label='End agente', s=60)
    plt.title(f"{tag.capitalize()} - Episodio {episode}")
    plt.xlabel("Step temporale")
    plt.ylabel("Valore (es. angolo)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, f"{tag}_ep{episode}.png"))
    plt.close()

def train_ddpg(env=None, num_episodes=10001):
    if env is None:
        env = TrackingEnv()
    state_dim = 2
    action_dim = 1
    agent = DDPGAgent(state_dim, action_dim)
    reward_history, success_history = [], []
    counter = 0
    tolerance = 0.01

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        real_state = torch.tensor(state, dtype=torch.float32)
        state = torch.tensor(state, dtype=torch.float32)

        state = state.clone()
        state[1:] += torch.normal(mean=0.0, std=0.003, size=(1,), device=state.device)

        agent.noise_std = max(agent.min_noise_std, agent.noise_std * agent.noise_decay)     # Exploration
        trajectory, target_trajectory = [], []
        attached_counter = 0
        total_attached_counter = 0

        while not done:
            trajectory.append(state[0].detach().numpy())
            target_trajectory.append(state[1].detach().numpy())
            action = agent.actor(state).detach().numpy()
            noise = np.random.normal(0, agent.noise_std, size=action.shape)
            noisy_action = action + noise   # Exploration
            noisy_action = np.clip(noisy_action, env.action_space.low, env.action_space.high)
            action_tensor = torch.tensor(noisy_action, dtype=torch.float32)

            next_state, _, done, truncated, _ = env.step(noisy_action)
            real_next_state = torch.tensor(next_state, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            

            next_state = next_state.clone()
            next_state[1:] += torch.normal(mean=0.0, std=0.003, size=(1,), device=next_state.device)


            if torch.norm(real_next_state[0] - real_state[1]) < tolerance:
                total_attached_counter += 1
                attached_counter += 1
            else:
                attached_counter = 0

            #reward = agent.reward_function(real_state, action_tensor, real_next_state, tolerance)
            reward = agent.reward_function(state, action_tensor, next_state, tolerance)

            if attached_counter > 20 or truncated or (total_attached_counter > 0 and torch.norm(real_next_state[0] - real_state[1]) > tolerance):
                done = True
            
            transition = (state.numpy(), action_tensor.numpy(), reward, next_state.numpy(), float(done))
            agent.buffer.push(transition)
            if len(agent.buffer) > 1000:
                agent.update()
            state = next_state
            real_state = real_next_state
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
        if episode % CHECKPOINT_INTERVAL == 0 and episode > 0:
            save_checkpoint(agent, episode)
        if episode % 50 == 0 and episode > 0:
            save_trajectory_plot(trajectory, target_trajectory, episode)

        if len(reward_history) > EARLY_STOPPING_EPISODES and np.mean(reward_history[-EARLY_STOPPING_EPISODES:]) > 2000:
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
