import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset
from ddpg_rot_dyn import DDPGAgent
from env_rot import TrackingEnv

# ==== DEFINIZIONE DEL MODELLO (es. MLP per continui) ====
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ==== FUNZIONE DI TRAINING PER BEHAVIORAL CLONING ====
def train_model(model, observations, actions, epochs=10, batch_size=64):
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
def dagger(env, expert_model, agent_model, initial_obs, initial_act, iterations=10, episodes_per_iter=5):
    observations = list(initial_obs)
    actions = list(initial_act)

    # Allena il modello iniziale con BC
    print("[INFO] Inizio training BC iniziale")
    train_model(agent_model, observations, actions, epochs=20)

    # Inizia il loop DAgger
    for it in range(iterations):
        print(f"[INFO] Iterazione DAgger {it+1}/{iterations}")

        new_obs = []
        new_act = []

        for _ in range(episodes_per_iter):
            obs = env.reset()
            done = False
            steps = 0
            episode_obs = []
            episode_act = []

            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action = agent_model(obs_tensor).squeeze(0).numpy()
                obs, _, done, _ = env.step(action)
                steps += 1

                with torch.no_grad():
                    expert_action = expert_model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()

                episode_obs.append(obs)
                episode_act.append(expert_action)

            # Aggiungi solo se episodio >= 90 step (comportamento ritenuto ottimale)
            if steps >= 90:
                new_obs.extend(episode_obs)
                new_act.extend(episode_act)

        # Aggrega i nuovi dati
        observations.extend(new_obs)
        actions.extend(new_act)

        # Ri-allena il modello
        train_model(agent_model, observations, actions, epochs=10)

    # Salva il modello finale
    torch.save(agent_model.state_dict(), "IL/dagger_model_rot_0.5_0.01.pth")
    print("[INFO] Modello DAgger salvato.")

# ==== ESECUZIONE ====
if __name__ == "__main__":
    from envs import TrackingEnv  # O il tuo ambiente custom

    # Parametri dell'ambiente
    input_dim = 2    # Modifica per rotazione (es. 2) o traslazione (es. 4)
    output_dim = 1   # Modifica per rotazione (es. 1) o traslazione (es. 2)

    # Istanzia ambiente ed esperto
    env = TrackingEnv()

    expert_model = DDPGAgent(input_dim, output_dim)
    expert_model.load_state_dict(torch.load("Rotazioni-dinamiche/No-noise/ddpg_mov_0.01_20250509_163508/checkpoint_ep782.pth"))
    expert_model.eval()

    agent_model = PolicyNetwork(input_dim, output_dim)

    # Carica dati esperti
    with open("trajectories/dataset_rot.npz", "rb") as f:
        expert_data = pickle.load(f)
    initial_obs, initial_act = zip(*expert_data)

    dagger(env, expert_model, agent_model, initial_obs, initial_act, iterations=10, episodes_per_iter=5)
