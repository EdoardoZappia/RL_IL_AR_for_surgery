import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class TrajectoryDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.observations = torch.tensor(data["observations"], dtype=torch.float32)
        self.actions = torch.tensor(data["actions"], dtype=torch.float32)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]

dataset = TrajectoryDataset("trajectories_correct/dataset_transl_std_0.003.npz")
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

class BCModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BCModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, output_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) * 5.0
        return action

def train_bc_model(model, dataloader, num_epochs=200, learning_rate=1e-3, device=None, save_path="IL/BC_dataset_correct/bc_policy_transl_0.2_0.05_std_0.003.pth"):

    # Crea la cartella se non esiste
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0

        for obs_batch, act_batch in dataloader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            pred_actions = model(obs_batch)
            loss = criterion(pred_actions, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * obs_batch.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"Modello salvato in '{save_path}'")

if __name__ == "__main__":
    model = BCModel(input_dim=4, output_dim=2)
    train_bc_model(model, dataloader)