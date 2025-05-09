import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from mujoco import MjModel, MjData
import torch

# Carica il modello MuJoCo da file
MODEL_PATH = "ellipsoid_rototranslation_2D.xml"  # Assicurati che il file esista nella directory corretta

class TrackingEnv(gym.Env):
    """Ambiente Gymnasium per il tracking del target in 2D con rotazione"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()

        # Carica il modello MuJoCo da file
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)

        # Definizione dello spazio delle velocità dx e dy
        self.action_space = spaces.Box(low=np.array([-5]), high=np.array([5]), dtype=np.float32)

        # Definizione dello spazio delle osservazioni [theta, theta_target]
        obs_low = np.array([-3.14, -3.14], dtype=np.float32)
        obs_high = np.array([3.14, 3.14], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Render
        self.render_mode = render_mode
        self.renderer = None
        self.step_counter = 0
        #self.max_steps = 400   per target statico
        self.max_steps = 100  # per target dinamico

    def step(self, action):
        """Esegue un passo nel simulatore MuJoCo"""
        # Converti il tensor di PyTorch in NumPy
        self.step_counter += 1
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        self.data.qvel[2] = action
        
        mujoco.mj_step(self.model, self.data)

        # ROTAZIONI
        theta = np.random.uniform(-0.01, 0.01)  # angolo di rotazione casuale
        proposed_theta = self.data.qpos[5] + theta
        proposed_theta = torch.tensor(proposed_theta, dtype=torch.float32)

        if proposed_theta >= -3.14 and proposed_theta <= 3.14:
            self.data.qpos[5] = proposed_theta

        obs = np.array([self.data.qpos[2], self.data.qpos[5]])  # theta, theta_target

        # **Reward NON viene calcolato qui** (sarà compito del modello di RL)
        reward = 0.0  

        # L'ambiente **non termina mai** da solo, spetta al modello RL decidere
        done = False
        truncated = False

        if self.step_counter >= self.max_steps:
            truncated = True

        return obs, reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        """Resetta l'ambiente"""
        super().reset(seed=seed)

        self.step_counter = 0

        # Resetta la simulazione
        mujoco.mj_resetData(self.model, self.data)

        #self.data.qpos[:] = np.random.uniform(low=-0.6, high=0.6, size=(6,))  # Posizione casuale dell'agente e del target
        self.data.qpos[:] = np.random.uniform(low=-0.5, high=0.5, size=(6,))  # Posizione casuale dell'agente e del target
        #self.data.qpos[:] = np.random.uniform(low=-0.01, high=0.01, size=(6,))  # Agente parte dentro la tolleranza
        
        obs = np.array([self.data.qpos[2], self.data.qpos[5]])
        return obs, {}

    def render(self):
        """Renderizza la simulazione"""
        if self.render_mode == "human":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model)
            self.renderer.update_scene(self.data)
            self.renderer.render()
        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, width=500, height=500)
            self.renderer.update_scene(self.data)
            return self.renderer.render()

    def close(self):
        """Chiude il simulatore"""
        if self.renderer is not None:
            self.renderer = None
