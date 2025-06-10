import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from imitation.data.types import Trajectory
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env
from imitation.algorithms.mce_irl import MCEIRL
from imitation.data import rollout

from env_rot import TrackingEnv

# ======== CONFIG ========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPISODE_LEN = 100
REWARD_SAVE_PATH = "IL/MCE/reward_model.pt"
# ========================

# Carica il dataset esperto
expert_data = np.load("trajectories/dataset_rot.npz")
observations = expert_data["observations"]
actions = expert_data["actions"]

# Ricostruisci le traiettorie: un episodio ogni 100 step
n_episodes = len(observations) // EPISODE_LEN
trajectories = []
for i in range(n_episodes):
    start = i * EPISODE_LEN
    end = (i + 1) * EPISODE_LEN
    obs_traj = observations[start:end]
    acts_traj = actions[start:end]
    trajectories.append(Trajectory(obs=obs_traj, acts=acts_traj, infos=None, terminal=True))

# Crea l'ambiente
venv = make_vec_env(
    lambda: TrackingEnv(),
    n_envs=1,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
)

# Inizializza e allena MCE IRL
irl = MCEIRL(demos=trajectories, venv=venv, deterministic_policy=False)
irl.train()

# Salva il modello della reward
torch.save(irl.reward_net.state_dict(), REWARD_SAVE_PATH)
print(f"Modello della reward salvato in {REWARD_SAVE_PATH}")
