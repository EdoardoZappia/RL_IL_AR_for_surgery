import torch
import numpy as np
from env_rot import TrackingEnv
from ddpg_rot_dyn_DME import DDPGAgent
import ddpg_rot_dyn_DME

# Carica dati
data = np.load("trajectories/dataset_rot.npz")
obs = data['observations']  # shape = (N, 2)
actions = data['actions']   # shape = (N, 1)

# Numero di step per episodio
steps_per_episode = 100

# Divisione in episodi
obs_episodes = obs.reshape(-1, steps_per_episode, obs.shape[1])       # (num_episodes, 100, 2)
actions_episodes = actions.reshape(-1, steps_per_episode, actions.shape[1])  # (num_episodes, 100, 1)


class RewardNet(torch.nn.Module):
    def __init__(self, input_dim=3, output_dim=1):
        super(RewardNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MaxEntIRL(torch.nn.Module):
    def __init__(self, reward_net, env, agent):
        super(MaxEntIRL, self).__init__()
        self.env = env
        self.reward_net = reward_net
        self.optimizer = torch.optim.Adam(self.reward_net.parameters(), lr=0.001)
        self.batch_size = 64
        self.agent = agent

    def compute_reward(self, obs, actions):
        inputs = torch.cat((obs, actions), dim=-1)  # Concatenate observations and actions
        return self.reward_net(inputs)

    def compute_mean_reward(self, obs, actions):
        rewards = self.compute_reward(obs, actions)
        return rewards.mean()

    def compute_mean_expert_reward(self, obs_expert, actions_expert):
        rewards = self.compute_reward(obs_expert, actions_expert)
        return rewards.mean()

    def loss_function(self, obs_policy, actions_policy, obs_expert, actions_expert):
        mean_reward_policy = self.compute_mean_reward(obs_policy, actions_policy)
        mean_reward_expert = self.compute_mean_reward(obs_expert, actions_expert)
        return -(mean_reward_expert - mean_reward_policy)

    def generate_policy_trajectory(self, batch_size=64):
        states_list = []
        actions_list = []

        for ep in range(batch_size):
            ep_states, ep_actions = [], []
            state, _ = self.env.reset()
            done = False

            while not done:
                # Prepara il tensore stato
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    action = self.agent.actor(state_tensor).squeeze(0).numpy()

                # Salva stato e azione
                ep_states.append(state_tensor.squeeze(0).numpy())
                ep_actions.append(np.array(action))

                # Step ambiente
                next_state, reward, _, truncated, _ = self.env.step(action)
                done = truncated
                state = next_state

            states_list.append(np.array(ep_states))
            actions_list.append(np.array(ep_actions))

        states_tensor = torch.tensor(np.array(states_list), dtype=torch.float32)
        actions_tensor = torch.tensor(np.array(actions_list), dtype=torch.float32)

        return states_tensor, actions_tensor

    def save_checkpoint(agent, reward_history, success_history):
        path = os.path.join(RUN_DIR, f"checkpoint_ep{episode}.pth")
        torch.save({
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            'actor_target_state_dict': agent.actor_target.state_dict(),
            'critic_target_state_dict': agent.critic_target.state_dict(),
            'optimizer_actor_state_dict': agent.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': agent.optimizer_critic.state_dict(),
            'replay_buffer': agent.buffer,
            'reward_history': reward_history,
            'success_history': success_history,
            'noise_std': agent.noise_std,
        }, path)

    def load_checkpoint(path, agent):
        checkpoint = torch.load(path)
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        agent.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        agent.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        agent.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
        agent.buffer = checkpoint['replay_buffer']
        agent.noise_std = checkpoint['noise_std']
        return checkpoint['reward_history'], checkpoint['success_history']



    def train(self, obs_episodes, actions_episodes, epochs=2000, steps_per_episode=100):
        obs_expert_all = torch.tensor(obs_episodes, dtype=torch.float32)      # (N_ep, T, obs_dim)
        actions_expert_all = torch.tensor(actions_episodes, dtype=torch.float32)  # (N_ep, T, act_dim)

        for epoch in range(epochs):
            # 1. Campiona batch esperto
            idx = np.random.choice(obs_expert_all.shape[0], self.batch_size, replace=False)
            obs_expert = obs_expert_all[idx].reshape(-1, obs_expert_all.shape[2])
            actions_expert = actions_expert_all[idx].reshape(-1, actions_expert_all.shape[2])

            # 2. Genera batch policy
            obs_policy, actions_policy = self.generate_policy_trajectory(batch_size=self.batch_size)
            obs_policy = obs_policy.reshape(-1, obs_policy.shape[2])
            actions_policy = actions_policy.reshape(-1, actions_policy.shape[2])

            # 3. Calcola loss e ottimizza
            loss = self.loss_function(obs_policy, actions_policy, obs_expert, actions_expert)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 4. Logging
            if (epoch + 1) % 10 == 0:
                r_exp = self.compute_mean_reward(obs_expert, actions_expert).item()
                r_pol = self.compute_mean_reward(obs_policy, actions_policy).item()
                print(f"[Epoch {epoch+1}] Loss: {loss.item():.4f} | R_exp: {r_exp:.4f} | R_pol: {r_pol:.4f}")
            
            if epoch % 100 == 0:
                agent = train_ddpg(self.reward_net, num_episodes=200, checkpoint_path="IL/checkpoint_DME_DDPG.pth")
            
        torch.save(self.reward_net.state_dict(), "IL/DME_rot_reward_net.pth")
        print("Rete di reward salvata in 'reward_net.pth'")

        print("\n--- Valutazione reward appresa ---")
        tolerance = 0.01
        steps_per_episode = obs_expert_all.shape[1]
        device = next(self.reward_net.parameters()).device

        # 5 episodi esperti
        print("\n[Episodi esperti]")
        for i in [1, 200, 400, 600, 800, 1000]:
            obs_ep = obs_expert_all[i]
            act_ep = actions_expert_all[i]
            inputs = torch.cat([obs_ep, act_ep], dim=-1).to(device)

            with torch.no_grad():
                rewards = self.reward_net(inputs).squeeze(-1)
                total_reward = rewards.sum().item()

            theta_next = obs_ep[1:, 0]
            theta_target = obs_ep[:-1, 1]
            attached = torch.abs(theta_next - theta_target) < tolerance
            attached_count = attached.sum().item()

            print(f"[Esperto {i}] Reward totale: {total_reward:.2f} | Attaccato: {attached_count}/{steps_per_episode - 1}")

        # 5 episodi random
        print("\n[Episodi random]")
        env = self.env
        for i in [1, 200, 400, 600, 800, 1000]:
            ep_obs, ep_actions = [], []
            state, _ = env.reset()
            done = False

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action = self.agent.actor(state_tensor).squeeze(0).numpy()
                ep_obs.append(state_tensor.numpy())
                ep_actions.append(action)
                next_state, _, _, truncated, _ = env.step(action)
                done = truncated
                state = next_state

            obs_ep = torch.tensor(np.array(ep_obs), dtype=torch.float32)
            act_ep = torch.tensor(np.array(ep_actions), dtype=torch.float32)

            inputs = torch.cat([obs_ep, act_ep], dim=-1).to(device)
            with torch.no_grad():
                rewards = self.reward_net(inputs).squeeze(-1)
                total_reward = rewards.sum().item()

            theta_next = obs_ep[1:, 0]
            theta_target = obs_ep[:-1, 1]
            attached = torch.abs(theta_next - theta_target) < tolerance
            attached_count = attached.sum().item()

            print(f"[Random {i}] Reward totale: {total_reward:.2f} | Attaccato: {attached_count}/{len(theta_target)}")


if __name__ == "__main__":
    # Inizializza ambiente
    env = TrackingEnv(render_mode=None)

    # Inizializza rete di reward
    reward_net = RewardNet(input_dim=3, output_dim=1)

    agent = DDPGAgent(2, 1)

    # Inizializza MaxEnt IRL
    maxent_irl = MaxEntIRL(reward_net, env, agent)

    # Addestra il modello
    maxent_irl.train(obs_episodes, actions_episodes, epochs=2000, steps_per_episode=100)