import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from mujoco import MjModel, MjData
import torch

# Carica il modello MuJoCo da file
MODEL_PATH = "ellipsoid_translation_2d.xml"  # Assicurati che il file esista nella directory corretta

class TrackingEnv(gym.Env):
    """Ambiente Gymnasium per il tracking del target in 2D con rotazione"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()

        # Carica il modello MuJoCo da file
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)

        # Definizione dello spazio delle azioni Fx e Fy
        #self.action_space = spaces.Box(low=np.array([-10, -10]), high=np.array([10, 10]), dtype=np.float32)

        # Definizione dello spazio delle velocità dx e dy
        self.action_space = spaces.Box(low=np.array([-5, -5]), high=np.array([5, 5]), dtype=np.float32)

        # Definizione dello spazio delle osservazioni [x, y, x_target, y_target]
        obs_low = np.array([-5, -5, -5, -5], dtype=np.float32)
        obs_high = np.array([5, 5, 5, 5], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Render
        self.render_mode = render_mode
        self.renderer = None
        self.step_counter = 0
        #self.max_steps = 400   per target statico
        self.max_steps = 100  # per target dinamico

    def rimbalzo(self):
        # Ottieni la posizione attuale
        pos = self.data.qpos[:2]
        vel = self.data.qvel[:2]

        # Limiti dell'area
        x_min, x_max = -2.0, 2.0    # -2.0 < x < 2.0
        y_min, y_max = -2.0, 2.0    # -2.0 < y < 2.0

        rimbalzato = False

        # Rimbalzo su X
        if pos[0] < x_min:
            pos[0] = x_min
            vel[0] *= -1
            rimbalzato = True
        elif pos[0] > x_max:
            pos[0] = x_max
            vel[0] *= -1
            rimbalzato = True

        # Rimbalzo su Y
        if pos[1] < y_min:
            pos[1] = y_min
            vel[1] *= -1
            rimbalzato = True
        elif pos[1] > y_max:
            pos[1] = y_max
            vel[1] *= -1
            rimbalzato = True

        # Applica le modifiche
        self.data.qpos[:2] = pos
        self.data.qvel[:2] = vel

        return rimbalzato


    def step(self, action):
        """Esegue un passo nel simulatore MuJoCo"""
        # Converti il tensor di PyTorch in NumPy
        self.step_counter += 1
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        #self.data.ctrl[:2] = action     # muovo solo l'agente
        self.data.qvel[:2] = action  # muovo solo l'agente
        mujoco.mj_step(self.model, self.data)

        target_pos = self.data.qpos[2:4]

        # Movimento casuale vincolato in un cerchio di raggio 1
        movement = np.random.uniform(low=-0.01, high=0.01, size=2)  # spostamento casuale
        proposed_pos = target_pos + movement
        proposed_pos = torch.tensor(proposed_pos, dtype=torch.float32)

        # Calcola distanza dalla posizione iniziale
        displacement = proposed_pos - self.target_center
        if np.linalg.norm(displacement) <= 1 and proposed_pos[0] >= -2 and proposed_pos[0] <= 2 and proposed_pos[1] >= -2 and proposed_pos[1] <= 2:
            self.data.qpos[2:4] = proposed_pos  # accetta lo spostamento
        # else: nessun movimento (rimane fermo)

        # # Movimento rettilineo
        # self.data.qpos[2:4] += self.target_velocity

        # proposed_pos = self.data.qpos[2:4]
        # proposed_pos = torch.tensor(proposed_pos, dtype=torch.float32)

        # # (facoltativo) Mantieni il target entro un'area
        # displacement = proposed_pos - self.target_center
        # if np.linalg.norm(displacement) > 0.1: # raggio di 0.1
        #     self.target_velocity *= -1  # inverte direzione quando esce dal raggio



        # qpos contiene tutti i joint (assumiamo 2 dell'agente + 2 del target)
        # [x_agent, y_agent, x_target, y_target]
        qpos = self.data.qpos

        x_agent, y_agent, x_target, y_target = qpos[:4]

        #obs = np.array([x_agent, y_agent, x_target, y_target, self.step_counter], dtype=np.float32)
        obs = np.array([x_agent, y_agent, x_target, y_target], dtype=np.float32)

        # **Reward NON viene calcolato qui** (sarà compito del modello di RL)
        reward = 0.0  

        # L'ambiente **non termina mai** da solo, spetta al modello RL decidere
        done = False
        truncated = False

        if self.step_counter >= self.max_steps:
            truncated = True

        rimbalzato = self.rimbalzo()

        return obs, reward, done, truncated, {}, rimbalzato

    def reset(self, seed=None, options=None):
        """Resetta l'ambiente"""
        super().reset(seed=seed)

        self.step_counter = 0

        # Resetta la simulazione
        mujoco.mj_resetData(self.model, self.data)

        #self.data.qpos[:] = np.random.uniform(low=-0.6, high=0.6, size=(4,))  # Posizione casuale dell'agente e del target
        self.data.qpos[:] = np.random.uniform(low=-0.2, high=0.2, size=(4,))  # Posizione casuale dell'agente e del target
        #self.data.qpos[:] = np.random.uniform(low=-0.01, high=0.01, size=(4,))  # Agente parte dentro la tolleranza
        self.target_center = self.data.qpos[2:4]  # Posizione iniziale del target
        # # Stato iniziale [0,0] per l'agente [1.0, 0.5] per il target
        # if target is None:
        #     target = [0.3, -0.5]

        # self.data.qpos[:2] = [0, 0]  # Posizione dell'agente
        # self.data.qpos[2:4] = target  # Posizione del target

        # self.target_center = target
        #self.target_velocity = np.array([0.01, 0.0])  # Velocità del target per movimento rettilineo

        # theta = np.random.uniform(0, 2 * np.pi)
        # speed = 0.005  # puoi regolarla
        # self.target_velocity = np.array([
        #         speed * np.cos(theta),
        #         speed * np.sin(theta)
        #         ], dtype=np.float32)


        obs = self.data.qpos

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
