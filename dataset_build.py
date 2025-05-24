import os
import torch
import numpy as np
from rototransl_env import TrackingEnv
from ddpg_dyn import DDPGAgent as DDPGTranslAgent
from ddpg_rot_dyn import DDPGAgent as DDPGRotAgent

def load_agents(checkpoint_path_transl, checkpoint_path_rot, env=None):
    if env is None:
        env = TrackingEnv()

    state_dim_trasl = 4  # x, y, x_target, y_target
    state_dim_rot = 2    # theta, theta_target

    agent_transl = DDPGTranslAgent(state_dim_trasl, 2)
    agent_rot = DDPGRotAgent(state_dim_rot, 1)

    ckpt_transl = torch.load(checkpoint_path_transl, map_location=torch.device('cpu'))
    ckpt_rot = torch.load(checkpoint_path_rot, map_location=torch.device('cpu'))

    agent_transl.actor.load_state_dict(ckpt_transl['actor_state_dict'])
    agent_rot.actor.load_state_dict(ckpt_rot['actor_state_dict'])

    agent_transl.eval()
    agent_rot.eval()

    return agent_transl, agent_rot

def test_dual_agents(agent_transl, agent_rot, env=None, num_episodes=1001, tolerance_transl=0.02, tolerance_rot=0.01):
    if env is None:
        env = TrackingEnv()

    all_obs = []
    all_actions = []
    saved_counter = 0

    for ep in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        #state[3:5] += torch.normal(mean=0.0, std=0.005, size=(2,), device=state.device)
        #state[5:] += torch.normal(mean=0.0, std=0.001, size=(1,), device=state.device)

        done = False
        attached_counter = 0
        total_attached_counter = 0
        total_attached_counter_transl = 0
        total_attached_counter_rot = 0

        episode_obs = []
        episode_actions = []

        while not done:
            with torch.no_grad():
                state_pos = torch.cat([state[:2], state[3:5]], dim=0)
                state_rot = torch.cat([state[2:3], state[5:6]], dim=0)
                action_xy = agent_transl.actor(state_pos)
                action_rot = agent_rot.actor(state_rot)

            action = torch.cat([action_xy, action_rot], dim=0).detach().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)

            episode_obs.append(state.detach().numpy())
            episode_actions.append(action.copy())

            next_state, _, done, truncated, _, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)

            #next_state[3:5] += torch.normal(mean=0.0, std=0.005, size=(2,), device=next_state.device)
            #next_state[5:] += torch.normal(mean=0.0, std=0.001, size=(1,), device=next_state.device)

            dist_transl = torch.norm(next_state[:2] - state[3:5])
            dist_rot = torch.abs(next_state[2] - state[5])

            if dist_transl < tolerance_transl:
                total_attached_counter_transl += 1

            if dist_rot < tolerance_rot:
                total_attached_counter_rot += 1

            if dist_transl < tolerance_transl and dist_rot < tolerance_rot:
                total_attached_counter += 1
                attached_counter += 1
            else:
                attached_counter = 0

            done = truncated
            state = next_state

        if total_attached_counter > 90:
            all_obs.extend(episode_obs)
            all_actions.extend(episode_actions)
            saved_counter += 1
            print(f"[Episode {ep}] SALVATO ({total_attached_counter} attached)")
        else:
            print(f"[Episode {ep}] scartato ({total_attached_counter} attached)")

    env.close()

    if all_obs:
        os.makedirs("trajectories", exist_ok=True)
        np.savez("trajectories/dataset_filtered.npz",
                 observations=np.array(all_obs),
                 actions=np.array(all_actions))
        print(f"\nDataset salvato con {len(all_obs)} passi totali da {saved_counter} episodi validi")
    else:
        print("\nNessun episodio valido, dataset non salvato.")

# --- MAIN ---
if __name__ == "__main__":
    ckpt_transl = "RL_IL_AR_for_surgery/Traslazioni-dinamiche/No-noise/ddpg_run_dyn20250503_160754/checkpoint_ep2930.pth"
    ckpt_rot = "RL_IL_AR_for_surgery/Rotazioni-dinamiche/No-noise/ddpg_mov_0.01_20250509_163508/checkpoint_ep782.pth"
    agent_transl, agent_rot = load_agents(ckpt_transl, ckpt_rot)
    test_dual_agents(agent_transl, agent_rot, num_episodes=1000)
