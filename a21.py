# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 17:25:06 2025

Author: 陈东杰
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from gymnasium.vector import SyncVectorEnv

# ========= Config =========
GAMMA = 0.99
LEARNING_RATE = 1e-3
TOTAL_STEPS = 1_000_000
NUM_REPEATS = 5
NUM_ENVS = 64
BATCH_SIZE = 64
MOVING_AVG_WINDOW = 200
MAX_EPISODE_LENGTH = 500  # Fixed to reward overflow

# ========= Environment Setup =========
def make_env():
    return gym.make("CartPole-v1")

envs = SyncVectorEnv([make_env for _ in range(NUM_ENVS)])
obs_dim = envs.single_observation_space.shape[0]
action_dim = envs.single_action_space.n

# ========= Networks =========
class PolicyNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(action_dim, activation='softmax')

    def call(self, x):
        x = self.fc1(x)
        return self.fc2(x)

class ValueNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.v = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.fc1(x)
        return self.v(x)

# ========= Helper =========
def discount_rewards(rewards, gamma=0.99):
    discounted = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = rewards[t] + gamma * running_add
        discounted[t] = running_add
    return discounted

def moving_average(data, window_size=200):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# ========= Training =========
def train_pg(envs, algo="reinforce", run_id=1):
    policy_net = PolicyNetwork()
    value_net = ValueNetwork() if algo in ["ac", "a2c"] else None

    policy_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    value_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    episode_rewards = []
    episode_steps = []
    log_data = []
    env_rewards = np.zeros(NUM_ENVS)
    episode_lengths = np.zeros(NUM_ENVS, dtype=np.int32)
    completed_episodes = 0

    state, _ = envs.reset()
    current_steps = 0
    trajectories = [[] for _ in range(NUM_ENVS)]

    while current_steps < TOTAL_STEPS:
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        probs = policy_net(state_tensor).numpy()
        actions = [np.random.choice(action_dim, p=p) for p in probs]
        next_state, rewards, dones, _, _ = envs.step(actions)

        for i in range(NUM_ENVS):
            if episode_lengths[i] < MAX_EPISODE_LENGTH:
                # storing (state, action, reward, next_state, done)
                trajectories[i].append((state[i], actions[i], rewards[i], next_state[i], dones[i]))
                env_rewards[i] += rewards[i]
                episode_lengths[i] += 1

            if dones[i] or episode_lengths[i] >= MAX_EPISODE_LENGTH:
                traj = trajectories[i]
                if len(traj) > 0:
                    # Extract trajectory data with all five elements.
                    states = np.array([s for s, a, r, ns, d in traj], dtype=np.float32)
                    actions_arr = np.array([a for s, a, r, ns, d in traj], dtype=np.int32)
                    rewards_arr = np.array([r for s, a, r, ns, d in traj], dtype=np.float32)
                    next_states = np.array([ns for s, a, r, ns, d in traj], dtype=np.float32)
                    dones_arr = np.array([d for s, a, r, ns, d in traj], dtype=np.float32)
                    returns = discount_rewards(rewards_arr, GAMMA)

                    with tf.GradientTape() as tape:
                        logits = policy_net(states)
                        action_masks = tf.one_hot(actions_arr, action_dim)
                        log_probs = tf.math.log(tf.reduce_sum(logits * action_masks, axis=1) + 1e-8)

                        # ========= Policy Loss Calculation =========
                        if algo == "reinforce":
                            # REINFORCE: Use Monte Carlo returns directly.
                            loss = -tf.reduce_mean(log_probs * returns)
                        elif algo == "ac":
                            # Basic Actor-Critic: Use one-step TD target.
                            next_values = tf.squeeze(value_net(next_states))
                            targets = rewards_arr + GAMMA * (1 - dones_arr) * next_values
                            values = tf.squeeze(value_net(states))
                            advantages = targets - values
                            loss = -tf.reduce_mean(log_probs * advantages)
                        elif algo == "a2c":
                            # A2C: Use full Monte Carlo returns with advantage = return - V(s)
                            values = tf.squeeze(value_net(states))
                            advantages = returns - tf.stop_gradient(values)
                            loss = -tf.reduce_mean(log_probs * advantages)

                    grads = tape.gradient(loss, policy_net.trainable_variables)
                    policy_optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))

                    # ========= Value Network Update =========
                    if algo in ["ac", "a2c"]:
                        with tf.GradientTape() as v_tape:
                            values = tf.squeeze(value_net(states))
                            if algo == "ac":
                                v_loss = tf.reduce_mean(tf.square(targets - values))
                            else:
                                v_loss = tf.reduce_mean(tf.square(returns - values))
                        v_grads = v_tape.gradient(v_loss, value_net.trainable_variables)
                        value_optimizer.apply_gradients(zip(v_grads, value_net.trainable_variables))

                    episode_rewards.append(env_rewards[i])
                    episode_steps.append(current_steps)
                    avg_reward = np.mean(episode_rewards[-100:])
                    log_data.append({
                        "Episode": completed_episodes + 1,
                        "Episode_Return": env_rewards[i],
                        "Episode_Return_smooth": avg_reward,
                        "env_step": current_steps
                    })
                    print(f"Episode {completed_episodes + 1}, Step {current_steps}, Reward {env_rewards[i]:.2f}")
                    completed_episodes += 1

                new_state, _ = envs.reset()
                next_state[i] = new_state[i]
                env_rewards[i] = 0
                episode_lengths[i] = 0
                trajectories[i] = []

        state = next_state
        current_steps += np.sum(~np.array(dones))

    df_log = pd.DataFrame(log_data)
    df_log.to_csv(f"PG_{algo.upper()}_Run{run_id}.csv", index=False)
    return episode_rewards, episode_steps

# ========= Multi-run + Plot =========
def run_and_plot():
    algos = ["reinforce", "ac", "a2c"]
    step_grid = np.linspace(0, TOTAL_STEPS, num=100)
    plt.figure(figsize=(20, 12))

    for algo in algos:
        all_rewards, all_steps, all_logs = [], [], []
        for r in range(NUM_REPEATS):
            print(f"\n=== {algo.upper()} Run {r+1} ===")
            rewards, steps = train_pg(envs, algo=algo, run_id=r+1)
            all_rewards.append(rewards)
            all_steps.append(steps)
            df_log = pd.read_csv(f"PG_{algo.upper()}_Run{r+1}.csv")
            df_log["Run"] = r + 1
            all_logs.append(df_log)

        df_all_mode = pd.concat(all_logs, ignore_index=True)
        df_all_mode.to_csv(f"PG_AllLogs_{algo.upper()}.csv", index=False)

        reward_interpolated = []
        for rewards, steps in zip(all_rewards, all_steps):
            steps = np.array(steps)
            rewards = np.array(rewards)
            unique_steps, indices = np.unique(steps, return_index=True)
            unique_rewards = rewards[indices]
            interp_rewards = np.interp(step_grid, unique_steps, unique_rewards,
                                       left=unique_rewards[0], right=unique_rewards[-1])
            reward_interpolated.append(interp_rewards)

        reward_mean = np.mean(reward_interpolated, axis=0)
        reward_std = np.std(reward_interpolated, axis=0)
        reward_mean_smooth = moving_average(reward_mean, 5)
        reward_std_smooth = moving_average(reward_std, 5)
        step_grid_smooth = step_grid[:len(reward_mean_smooth)]

        plt.plot(step_grid_smooth, reward_mean_smooth, label=algo.upper())
        plt.fill_between(step_grid_smooth,
                         reward_mean_smooth - reward_std_smooth,
                         reward_mean_smooth + reward_std_smooth,
                         alpha=0.2)

    plt.xlabel("Step")
    plt.ylabel("Episode Reward")
    plt.title("Learning Curve Comparison For Learning Rate = 1e-5")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("PG_Comparison_Plot_Interpolated.png")
    plt.show()

if __name__ == "__main__":
    run_and_plot()
