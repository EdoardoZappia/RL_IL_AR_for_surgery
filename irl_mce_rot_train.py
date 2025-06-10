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
REWARD_SAVE_PATH = "IL/MCE/reward_model.pt"
# ========================

# Carica il dataset esperto
expert_data = np.load("trajectories/dataset_rot_101_obs_100_actions.npz")
observations = expert_data["observations"]
actions = expert_data["actions"]

# Ricostruisci le traiettorie: un episodio ogni 100 step
EPISODE_LEN = 100  # azioni per episodio
N_OBS_PER_EPISODE = EPISODE_LEN + 1

n_episodes = len(actions) // EPISODE_LEN
trajectories = []

for i in range(n_episodes):
    obs_start = i * N_OBS_PER_EPISODE
    obs_end = obs_start + N_OBS_PER_EPISODE
    act_start = i * EPISODE_LEN
    act_end = act_start + EPISODE_LEN

    obs_traj = observations[obs_start:obs_end]   # 101
    acts_traj = actions[act_start:act_end]       # 100

    if len(obs_traj) != len(acts_traj) + 1:
        print(f"Skipping malformed trajectory {i}: {len(obs_traj)} obs, {len(acts_traj)} acts")
        continue

    trajectories.append(Trajectory(obs=obs_traj, acts=acts_traj, infos=None, terminal=True))


rng = np.random.default_rng(seed=0)

# Crea l'ambiente
venv = make_vec_env(
    lambda: TrackingEnv(),
    rng=rng,
    n_envs=1,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
)

# Inizializza e allena MCE IRL
irl = MCEIRL(demos=trajectories, venv=venv, deterministic_policy=False)
irl.train()

# Salva il modello della reward
torch.save(irl.reward_net.state_dict(), REWARD_SAVE_PATH)
print(f"Modello della reward salvato in {REWARD_SAVE_PATH}")
