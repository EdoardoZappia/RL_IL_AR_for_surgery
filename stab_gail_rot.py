import numpy as np
import torch
import os
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.types import Trajectory
from imitation.data.wrappers import RolloutInfoWrapper
from env_rot import TrackingEnv

# ====== CONFIG ======
TRAJ_PATH = "trajectories/dataset_rot.npz"
POLICY_PATH = "models/gail_policy"
DISC_PATH = "models/gail_discriminator.pt"
TOTAL_STEPS = 100_000
EPISODE_LEN = 100  # lunghezza fissa di ogni episodio esperto
# ====================

# === 1. Carica dataset esperto e segmenta in Trajectory ===
print("Caricamento delle traiettorie esperte...")
data = np.load(TRAJ_PATH, allow_pickle=True)
obs = data["observations"]
acts = data["actions"]
num_episodes = len(obs) // EPISODE_LEN

expert_trajs = []
for i in range(num_episodes):
    start = i * EPISODE_LEN
    end = start + EPISODE_LEN

    last_obs = obs[end - 1 : end]  # ripeti ultima osservazione
    extended_obs = np.concatenate([obs[start:end], last_obs], axis=0)

    traj = Trajectory(
        obs=extended_obs,       # shape (101, obs_dim)
        acts=acts[start:end],   # shape (100, act_dim)
        terminal=True,
        infos=None
    )
    expert_trajs.append(traj)
print(f"  → Caricate {len(expert_trajs)} traiettorie")

# === 2. Setup ambiente ===
print("Inizializzazione ambiente...")
venv = DummyVecEnv([lambda: TrackingEnv()])
venv = RolloutInfoWrapper(venv)

# === 3. Inizializza PPO e GAIL ===
print("Inizializzazione GAIL...")
policy = PPO("MlpPolicy", venv, verbose=1)

gail_trainer = GAIL(
    demonstrations=expert_trajs,
    demo_batch_size=64,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=policy,
)

# === 4. Training ===
print("Inizio training GAIL...")
gail_trainer.train(TOTAL_STEPS)
print("Training completato!")

# === 5. Salvataggio ===
os.makedirs("models", exist_ok=True)
policy.save(POLICY_PATH)
print(f"Policy salvata in: {POLICY_PATH}")

torch.save(gail_trainer.reward_test.disc.state_dict(), DISC_PATH)
print(f"Discriminatore salvato in: {DISC_PATH}")
