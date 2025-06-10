import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.data.types import Trajectory
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.algorithms.mce_irl import MCEIRL

from env_rot import TrackingEnv

# ======== CONFIG ========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPISODE_LEN = 100
REWARD_SAVE_PATH = "IL/MCE/reward_model.pt"
DATA_PATH = "trajectories/dataset_rot_101_obs_100_actions.npz"
# ========================

# Carica il dataset esperto
expert_data = np.load(DATA_PATH)
observations = expert_data["observations"]
actions = expert_data["actions"]

# Ricostruisci le traiettorie (101 osservazioni, 100 azioni per episodio)
N_OBS_PER_EP = EPISODE_LEN + 1
n_episodes = len(actions) // EPISODE_LEN
trajectories = []

for i in range(n_episodes):
    obs_start = i * N_OBS_PER_EP
    obs_end = obs_start + N_OBS_PER_EP
    act_start = i * EPISODE_LEN
    act_end = act_start + EPISODE_LEN

    obs_traj = observations[obs_start:obs_end]
    acts_traj = actions[act_start:act_end]

    if len(obs_traj) != len(acts_traj) + 1:
        print(f"Skipping malformed trajectory {i}: {len(obs_traj)} obs, {len(acts_traj)} acts")
        continue

    trajectories.append(Trajectory(obs=obs_traj, acts=acts_traj, infos=None, terminal=True))

print(f"Caricate {len(trajectories)} traiettorie esperte.")

# Crea l'ambiente vettoriale
def make_env():
    env = TrackingEnv()
    env = RolloutInfoWrapper(env)
    return env

venv = DummyVecEnv([make_env])

# Inizializza MCE IRL
irl = MCEIRL(
    demonstrations=trajectories,
    env=venv,
)

# Sposta reward_net su GPU se disponibile
irl.reward_net.to(DEVICE)

# Allenamento
irl.train()

# Salva il modello della reward
torch.save(irl.reward_net.state_dict(), REWARD_SAVE_PATH)
print(f"Modello della reward salvato in {REWARD_SAVE_PATH}")
