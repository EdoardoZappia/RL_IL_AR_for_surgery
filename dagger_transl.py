import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from ddpg_rot_dyn import DDPGAgent
from environment import TrackingEnv

# ==== DEFINIZIONE DEL MODELLO (es. MLP per continui) ====
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x) * 5.0

# ==== FUNZIONE DI TRAINING PER BEHAVIORAL CLONING ====
def train_model(model, observations, actions, epochs=30, batch_size=64):
    dataset = TensorDataset(torch.tensor(observations, dtype=torch.float32),
                            torch.tensor(actions, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0
        for obs_batch, act_batch in dataloader:
            optimizer.zero_grad()
            predictions = model(obs_batch)
            loss = criterion(predictions, act_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# ==== LOOP PRINCIPALE DI DAGGER ====
def dagger(env, expert_model, agent_model, initial_obs, initial_act, iterations=15, episodes_per_iter=5):
    observations = list(initial_obs)
    actions = list(initial_act)

    # Tolleranze per comportamento "attached"
    tolerance_transl = 0.02

    # Allena il modello iniziale con BC
    print("[INFO] Inizio training BC iniziale")
    train_model(agent_model, observations, actions)

    # Inizia il loop DAgger
    for it in range(iterations):
        print(f"[INFO] Iterazione DAgger {it+1}/{iterations}")

        new_obs = []
        new_act = []

        for _ in range(episodes_per_iter):
            obs, _ = env.reset()
            state = torch.tensor(obs, dtype=torch.float32)

            state = state.clone()
            state[2:4] += torch.normal(mean=0.0, std=0.005, size=(2,), device=state.device)

            done = False
            episode_obs = []
            episode_act = []

            i = 0

            while not done:
                #state = torch.tensor(obs, dtype=torch.float32)

                obs_tensor = state.unsqueeze(0)
                with torch.no_grad():
                    action = agent_model(obs_tensor).squeeze(0).numpy()
                next_obs, _, done, truncated, _, _ = env.step(action)
                next_state = torch.tensor(next_obs, dtype=torch.float32)

                next_state = next_state.clone()
                next_state[2:4] += torch.normal(mean=0.0, std=0.005, size=(2,), device=next_state.device)

                done = truncated

                backup = env.get_state()

                with torch.no_grad():
                    expert_action = expert_model.actor(next_state.unsqueeze(0)).squeeze(0).numpy()
                
                virtual_next, _, _, _, _, _ = env.step(expert_action)
                virtual_next = torch.tensor(virtual_next, dtype=torch.float32)
                if torch.norm(virtual_next[:2]- next_state[2:4]) < tolerance_transl:
                    episode_obs.append(next_obs)
                    episode_act.append(expert_action)
                    i += 1
                env.set_state(backup)
                #obs = next_obs
                state = next_state
            
            print(f"Dataset aumentato {i} volte")
            new_obs.extend(episode_obs)
            new_act.extend(episode_act)

        # Aggrega i nuovi dati
        observations.extend(new_obs)
        actions.extend(new_act)

        # Ri-allena il modello
        train_model(agent_model, observations, actions)

    # Salva il modello finale
    torch.save(agent_model.state_dict(), "IL/dagger_model_transl_0.2_0.05_std_0.005.pth")
    print("[INFO] Modello DAgger salvato.")

def load_agents(checkpoint_path_rot, env=None):
    if env is None:
        env = TrackingEnv()

    state_dim_rot = 4    # theta, theta_target

    agent_transl = DDPGAgent(state_dim_rot, 2)

    ckpt_transl = torch.load(checkpoint_path_rot, map_location=torch.device('cpu'))

    agent_transl.actor.load_state_dict(ckpt_transl['actor_state_dict'])

    return agent_transl

# ==== ESECUZIONE ====
if __name__ == "__main__":

    # Parametri dell'ambiente
    input_dim = 4    # Modifica per rotazione (es. 2) o traslazione (es. 4)
    output_dim = 2   # Modifica per rotazione (es. 1) o traslazione (es. 2)

    # Istanzia ambiente ed esperto
    env = TrackingEnv()

    expert_model = load_agents("Traslazioni-dinamiche/Noisy/ddpg_run_dyn_mov_0.05_noisy_target_0.00520250504_150841/checkpoint_ep3639.pth", env)
    expert_model.actor.eval()

    agent_model = PolicyNetwork(input_dim, output_dim)

    # Carica dati esperti
    expert_data = np.load("trajectories/dataset_transl_std_0.005_0.001.npz")
    initial_obs = expert_data['observations']
    initial_act = expert_data['actions']

    dagger(env, expert_model, agent_model, initial_obs, initial_act)
