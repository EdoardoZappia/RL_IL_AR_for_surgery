import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from env_2_agents import TrackingEnv
import random
from collections import deque
import datetime

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

NUM_NEURONS_TRASL = 256
NUM_NEURONS_ROT = 128
LR_ACTOR_TRASL = 0.001     
LR_ACTOR_ROT = 0.001   
LR_CRITIC_TRASL = 0.001   
LR_CRITIC_ROT = 0.0008
GAMMA = 0.99
TAU = 0.005
EARLY_STOPPING_EPISODES = 30
CHECKPOINT_INTERVAL = 50

now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = f"runs/ddpg_rototransl_2_agents_dyn{now}"
os.makedirs(RUN_DIR, exist_ok=True)

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, NUM_NEURONS):
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
        return torch.tanh(self.fc3(x))

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, NUM_NEURONS):
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

class DDPGAgent_trasl(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = PolicyNet(state_dim, action_dim, NUM_NEURONS_TRASL)
        self.actor_target = PolicyNet(state_dim, action_dim, NUM_NEURONS_TRASL)
        self.critic = QNet(state_dim, action_dim, NUM_NEURONS_TRASL)
        self.critic_target = QNet(state_dim, action_dim, NUM_NEURONS_TRASL)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), LR_ACTOR_TRASL)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), LR_CRITIC_TRASL)
        self.buffer = ReplayBuffer(50000)
        self.batch_size = 128

    def reward_function(self, state, action, next_state, tolerance_transl, rimbalzato):
        pos = state[:2]
        target = state[2:4]              # target(t)
        next_pos = next_state[:2]        # agent(t+1)

        to_target = F.normalize(target - pos, dim=0)
        action_dir = F.normalize(action, dim=0)
        direction_reward = torch.dot(action_dir, to_target)
        direction_penalty = 1.0 - direction_reward

        reward = - 5 * direction_penalty

        if torch.norm(next_state[:2] - state[2:4]) < tolerance_transl:
            reward += 50

        if rimbalzato:
            reward -= 100
        
        return reward - 1

    def get_action(self, state):
        return self.actor(state) * 5.0

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

class DDPGAgent_rot(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = PolicyNet(state_dim, action_dim, NUM_NEURONS_ROT)
        self.actor_target = PolicyNet(state_dim, action_dim, NUM_NEURONS_ROT)
        self.critic = QNet(state_dim, action_dim, NUM_NEURONS_ROT)
        self.critic_target = QNet(state_dim, action_dim, NUM_NEURONS_ROT)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), LR_ACTOR_ROT)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), LR_CRITIC_ROT)
        self.buffer = ReplayBuffer(50000)
        self.batch_size = 128

    def reward_function(self, state, action, next_state, tolerance_rot):
        
        rot_error = torch.norm(state[1] - next_state[0])

        reward = - 3 * rot_error

        if torch.norm(next_state[0] - state[1]) < tolerance_rot:
            reward += 50
        
        return reward - 1

    def get_action(self, state):
        return self.actor(state)

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
        'actor_transl_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict()
    }, path)

def plot_and_save_rewards(rewards, title="Training Reward"):
    plt.figure()
    plt.plot(rewards, label='Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    # crea un nome file compatibile con il filesystem
    safe_title = title.lower().replace(" ", "_").replace("/", "_")
    filename = f"reward_plot_{safe_title}.png"
    plt.savefig(os.path.join(RUN_DIR, filename))
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

def train_ddpg(env=None, num_episodes=5001):
    if env is None:
        env = TrackingEnv()
    state_dim_trasl = 4
    state_dim_rot = 2
    agent_trasl = DDPGAgent_trasl(state_dim_trasl, 2)
    agent_rot = DDPGAgent_rot(state_dim_rot, 1)
    reward_history_rot, reward_history_trasl, reward_history_tot, success_history = [], [], [], []
    counter = 0
    tolerance_transl = 0.02
    tolerance_rot = 0.01
    noise_std = 0.5
    min_noise_std = 0.01
    noise_decay = 0.999

    for episode in range(num_episodes):
        state_trasl, state_rot = env.reset()
        done = False
        done_trasl = False
        done_rot = False
        total_reward = 0
        total_reward_trasl = 0
        total_reward_rot = 0
        real_state_trasl = torch.tensor(state_trasl, dtype=torch.float32)
        state_trasl = torch.tensor(state_trasl, dtype=torch.float32)
        real_state_rot = torch.tensor(state_rot, dtype=torch.float32)
        state_rot = torch.tensor(state_rot, dtype=torch.float32)

        # state = state.clone()
        # state[3:5] += torch.normal(mean=0.0, std=0.01, size=(2,), device=state.device)
        # state[5:] += torch.normal(mean=0.0, std=0.005, size=(1,), device=state.device)

        noise_std = max(min_noise_std, noise_std * noise_decay)
        trajectory, target_trajectory = [], []
        attached_counter = 0
        total_attached_counter = 0

        while not done:
            trajectory.append(state_trasl[:2].detach().numpy())
            target_trajectory.append(state_trasl[2:4].detach().numpy())

            action_trasl = agent_trasl.get_action(state_trasl).detach().numpy()
            action_rot = agent_rot.get_action(state_rot).detach().numpy()
            #action = np.concatenate((action_trasl, action_rot), axis=0)

            noise_trasl = np.random.normal(0, noise_std, size=action_trasl.shape)
            noisy_action_trasl = action_trasl + noise_trasl
            noisy_action_trasl = np.clip(noisy_action_trasl, env.action_space_trasl.low, env.action_space_trasl.high).squeeze()
            action_tensor_trasl = torch.tensor(noisy_action_trasl, dtype=torch.float32)

            noise_rot = np.random.normal(0, noise_std, size=action_rot.shape)
            noisy_action_rot = action_rot + noise_rot
            noisy_action_rot = np.clip(noisy_action_rot, env.action_space_rot.low, env.action_space_rot.high).squeeze(0)
            action_tensor_rot = torch.tensor(noisy_action_rot, dtype=torch.float32)

            noisy_action = np.concatenate((noisy_action_trasl, noisy_action_rot), axis=0)
            action_tensor = torch.tensor(noisy_action, dtype=torch.float32)

            next_state_trasl, next_state_rot, _, done, truncated, _, rimbalzato = env.step(noisy_action)
            real_next_state_trasl = torch.tensor(next_state_trasl, dtype=torch.float32)
            next_state_trasl = torch.tensor(next_state_trasl, dtype=torch.float32)
            real_next_state_rot = torch.tensor(next_state_rot, dtype=torch.float32)
            next_state_rot = torch.tensor(next_state_rot, dtype=torch.float32)

            # next_state = next_state.clone()
            # next_state[3:5] += torch.normal(mean=0.0, std=0.01, size=(2,), device=next_state.device)
            # next_state[5:] += torch.normal(mean=0.0, std=0.005, size=(1,), device=next_state.device)

            #if torch.norm(real_next_state[:2] - real_state[3:5]) < tolerance_transl:
            #    print("RAGGIUNTO TARGET, episodio:", episode)

            #if torch.norm(real_next_state[2] - real_state[5]) < tolerance_rot:
            #    print("RAGGIUNTO TARGET ROTAZIONE, episodio:", episode)

            if torch.norm(real_next_state_trasl[:2] - real_state_trasl[2:4]) < tolerance_transl and torch.norm(real_next_state_rot[0] - real_state_rot[1]) < tolerance_rot:
                #print(f"Episode: {episode} SUCCESSO")
                total_attached_counter += 1
                attached_counter += 1
            else:
                attached_counter = 0

            reward_trasl = agent_trasl.reward_function(state_trasl, action_tensor_trasl, next_state_trasl, tolerance_transl, rimbalzato)
            reward_rot = agent_rot.reward_function(state_rot, action_tensor_rot, next_state_rot, tolerance_rot)
            reward_tot = reward_trasl + reward_rot

            if torch.norm(real_next_state_trasl[:2] - real_state_trasl[2:4]) < tolerance_transl:
                done_trasl = True
            
            if torch.norm(real_next_state_rot[0] - real_state_rot[1]) < tolerance_rot:
                done_rot = True

            condition = torch.norm(real_next_state_trasl[:2] - real_state_trasl[2:4]) > tolerance_transl or torch.norm(real_next_state_rot[0] - real_state_rot[1]) > tolerance_rot
            if truncated or (total_attached_counter > 0 and condition) or attached_counter > 10:
                done = True

            transition_trasl = (state_trasl.numpy(), action_tensor_trasl.numpy(), reward_trasl, next_state_trasl.numpy(), float(done_trasl))
            transition_rot = (state_rot.numpy(), action_tensor_rot.numpy(), reward_rot, next_state_rot.numpy(), float(done_rot))

            agent_trasl.buffer.push(transition_trasl)
            agent_rot.buffer.push(transition_rot)

            if len(agent_trasl.buffer) > 1000:
                agent_trasl.update()
            if len(agent_rot.buffer) > 1000:
                agent_rot.update()

            state_trasl = next_state_trasl
            state_rot = next_state_rot
            real_state_trasl = real_next_state_trasl
            real_state_rot = real_next_state_rot

            total_reward_trasl += reward_trasl
            total_reward_rot += reward_rot
            total_reward += reward_tot

        if attached_counter > 10:
            counter += 1
            success_history.append(1)
            if counter % 10 == 0:
                save_trajectory_plot(trajectory, target_trajectory, episode, tag="success")
        else:
            success_history.append(0)

        reward_history_tot.append(total_reward)
        reward_history_trasl.append(total_reward_trasl)
        reward_history_rot.append(total_reward_rot)

        if total_attached_counter > 0:
            print(f"Episode {episode}, Reward tot: {total_reward:.2f}, Reward trasl: {total_reward_trasl:.2f}, Reward rot: {total_reward_rot:.2f}, Total attached counter: {total_attached_counter}, Successes: {counter}")

        #if episode % 10 == 0:
        #    print(f"Episode {episode}, Reward: {total_reward:.2f}, Attached_counter: {attached_counter}, Total attached counter: {total_attached_counter}, Successes: {counter}")
        if episode % CHECKPOINT_INTERVAL == 0 and episode > 0:
            save_checkpoint(agent_trasl, episode)
            save_checkpoint(agent_rot, episode)
        if episode % 50 == 0 and episode > 0:
            save_trajectory_plot(trajectory, target_trajectory, episode)

        #if len(reward_history) > EARLY_STOPPING_EPISODES and np.mean(reward_history[-EARLY_STOPPING_EPISODES:]) > 2000:
        if len(reward_history_tot) > EARLY_STOPPING_EPISODES and np.mean(success_history[-EARLY_STOPPING_EPISODES:]) == 1:
            print(f"Early stopping at episode {episode}")
            save_checkpoint(agent_trasl, episode)
            save_checkpoint(agent_rot, episode)
            save_trajectory_plot(trajectory, target_trajectory, episode)
            break

    np.save(os.path.join(RUN_DIR, 'rewards.npy'), reward_history_tot)
    np.save(os.path.join(RUN_DIR, 'successes.npy'), success_history)
    plot_and_save_rewards(reward_history_tot, title="Total Reward")
    plot_and_save_rewards(reward_history_trasl, title="Translational Reward")
    plot_and_save_rewards(reward_history_rot, title="Rotational Reward")
    env.close()
    return agent_trasl, agent_rot

if __name__ == "__main__":
    trained_agent_trasl, trained_agent_rot = train_ddpg()
