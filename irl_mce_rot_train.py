import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.data.types import Trajectory
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.algorithms.mce_irl import MCEIRL

from env_rot import TrackingEnv

# ======== CONFIG ========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPISODE_LEN = 100
REWARD_SAVE_PATH = "IL/MCE/reward_model.pt"
DATA_PATH = "trajectories/dataset_rot_101_obs_100_actions.npz"
SEED = 0
# ========================

# ======== NETWORK ========
class RewardNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        if action.ndim == 1:
            action = action.unsqueeze(1)
        x = torch.cat([state, action], dim=-1)
        return self.model(x)
# =========================

# ======== DATASET ========
expert_data = np.load(DATA_PATH)
observations = expert_data["observations"]
actions = expert_data["actions"]

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

print(f"✅ Caricate {len(trajectories)} traiettorie esperte.")
# =========================

# ======== AMBIENTE ========
def make_env():
    env = TrackingEnv()
    env = RolloutInfoWrapper(env)
    return env

venv = DummyVecEnv([make_env])
env = venv.envs[0]
# Patch per compatibilità con MCEIRL
if not hasattr(env, "state_dim"):
    env.state_dim = env.observation_space.shape[0]
# ==========================

# ======== RETE DI REWARD ========
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

reward_net = RewardNetwork(state_dim, action_dim).to(DEVICE)
# ================================

# ======== MCE IRL TRAINING ========
rng = np.random.default_rng(seed=SEED)

irl = MCEIRL(
    demonstrations=trajectories,
    env=venv,
    reward_net=reward_net,
    rng=rng,
)

irl.reward_net.to(DEVICE)
irl.train()
# ================================

# ======== SALVATAGGIO ========
torch.save(irl.reward_net.state_dict(), REWARD_SAVE_PATH)
print(f"Modello della reward salvato in: {REWARD_SAVE_PATH}")
# =============================
