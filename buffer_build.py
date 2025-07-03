import os
import torch
import torch.nn.functional as F
import numpy as np
from rototransl_env import TrackingEnv
from ddpg_dyn import DDPGAgent as DDPGTranslAgent
from ddpg_init_bc_rot import DDPGAgent as DDPGRotAgent

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

def reward_function_rot(state, action, next_state, tolerance):
        rot_error = torch.norm(state[1]-next_state[0])
        reward = - rot_error.item() * 3
        if torch.norm(next_state[0] - state[1]) < tolerance:
            reward += 100
        return reward - 1.0

def reward_function_transl(state, action, next_state, tolerance, rimbalzato):
        pos = state[:2]
        target = state[2:4]
        next_pos = next_state[:2]

        to_target = F.normalize(target - pos, dim=0)
        action_dir = F.normalize(action, dim=0)
        direction_reward = torch.dot(action_dir, to_target)
        direction_penalty = 1.0 - direction_reward

        reward = -5 * direction_penalty
        if torch.norm(next_state[:2] - state[2:4]) < tolerance:
            reward += 100
        if rimbalzato:
            reward -= 5

        return reward - 1.0

def test_dual_agents(agent_transl, agent_rot, env=None, num_episodes=10001, tolerance_transl=0.02, tolerance_rot=0.01):
    if env is None:
        env = TrackingEnv()

    reward_fn_rot = agent_rot.reward_function
    reward_fn_transl = agent_transl.reward_function

    all_obs_xy = []
    all_obs_rot = []
    saved_counter = 0
    max_transitions = 50000

    for ep in range(num_episodes):
        if len(all_obs_xy) >= max_transitions or len(all_obs_rot) >= max_transitions:
            break

        state, _ = env.reset()
        real_state = torch.tensor(state, dtype=torch.float32)
        state = torch.tensor(state, dtype=torch.float32)

        state = state.clone()
        state[3:5] += torch.normal(mean=0.0, std=0.003, size=(2,), device=state.device)
        state[5:] += torch.normal(mean=0.0, std=0.004, size=(1,), device=state.device)

        done = False
        attached_counter = 0
        total_attached_counter = 0
        total_attached_counter_transl = 0
        total_attached_counter_rot = 0

        episode_obs_xy = []
        episode_obs_rot = []

        while not done:
            with torch.no_grad():
                state_pos = torch.cat([state[:2], state[3:5]], dim=0)
                state_rot = torch.cat([state[2:3], state[5:6]], dim=0)
                action_xy = agent_transl.actor(state_pos)
                action_rot = agent_rot.actor(state_rot)

            action = torch.cat([action_xy, action_rot], dim=0).detach().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)

            s_pos = state_pos.detach().clone()
            s_rot = state_rot.detach().clone()

            next_state, _, done, truncated, _, rimbalzato = env.step(action)
            real_next_state = torch.tensor(next_state, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)

            next_state = next_state.clone()
            next_state[3:5] += torch.normal(mean=0.0, std=0.003, size=(2,), device=next_state.device)
            next_state[5:] += torch.normal(mean=0.0, std=0.004, size=(1,), device=next_state.device)

            next_pos = torch.cat([next_state[:2], next_state[3:5]], dim=0)
            next_rot = torch.cat([next_state[2:3], next_state[5:6]], dim=0)

            dist_transl = torch.norm(real_next_state[:2] - real_state[3:5])
            dist_rot = torch.abs(real_next_state[2] - real_state[5])

            if dist_transl < tolerance_transl:
                total_attached_counter_transl += 1
            if dist_rot < tolerance_rot:
                total_attached_counter_rot += 1
            if dist_transl < tolerance_transl and dist_rot < tolerance_rot:
                total_attached_counter += 1
                attached_counter += 1
            else:
                attached_counter = 0

            reward_rot = reward_function_rot(s_rot, action_rot, next_rot, tolerance_rot)
            reward_transl = reward_function_transl(s_pos, action_xy, next_pos, tolerance_transl, rimbalzato)

            transition_xy = (s_pos.numpy(), action_xy.numpy(), reward_transl, next_pos.numpy(), float(truncated))
            transition_rot = (s_rot.numpy(), action_rot.numpy(), reward_rot, next_rot.numpy(), float(truncated))

            episode_obs_xy.append(transition_xy)
            episode_obs_rot.append(transition_rot)

            done = truncated
            state = next_state
            real_state = real_next_state

        if total_attached_counter > 90:
            all_obs_xy.extend(episode_obs_xy)
            all_obs_rot.extend(episode_obs_rot)
            saved_counter += 1
            print(f"[Episode {ep}] SALVATO ({total_attached_counter} attached)")
        else:
            print(f"[Episode {ep}] scartato ({total_attached_counter} attached)")

        if len(all_obs_xy) >= max_transitions or len(all_obs_rot) >= max_transitions:
            print("\nRaggiunto limite massimo di 50000 transizioni. Interruzione anticipata.")
            break

    env.close()

    if all_obs_xy and all_obs_rot:
        os.makedirs("trajectories_correct", exist_ok=True)
        # np.savez("trajectories_correct/buffer_transitions_rot_std_0.004.npz",
        #          transitions=np.array(all_obs_rot[:max_transitions], dtype=object))
        np.savez("trajectories_correct/buffer_transitions_transl_std_0.003.npz",
                 transitions=np.array(all_obs_xy[:max_transitions], dtype=object))
        print(f"\nDataset salvato con {len(all_obs_xy[:max_transitions])} passi totali da {saved_counter} episodi validi")
        print(f"Dataset salvato con {len(all_obs_rot[:max_transitions])} passi totali da {saved_counter} episodi validi")
    else:
        print("\nNessun episodio valido, dataset non salvato.")

# --- MAIN ---
if __name__ == "__main__":
    ckpt_transl = "Traslazioni-dinamiche/Noisy/ddpg_run_dyn_mov_0.05_noisy_target_0.003_20250703_174313/checkpoint_ep4127.pth"
    ckpt_rot = "TEST_NOISE/Rotazioni-dinamiche/ddpg_mov_0.01_std_0.004_20250620_222306/checkpoint_ep2471.pth"
    agent_transl, agent_rot = load_agents(ckpt_transl, ckpt_rot)
    test_dual_agents(agent_transl, agent_rot)
