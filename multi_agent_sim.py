import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from collections import deque
import os
import random


class DualRobotEnv(gym.Env):
    def __init__(self):
        super(DualRobotEnv, self).__init__()
        self.layout_str = [
            'ØOØSOOØOOOØOOOOODOOO',
            'ØOØOOOØOOOØOOOOOOOOO',
            'ØOØOOOØOOOØOOOOOOOOO',
            'ØOØOOOØOOOØOOOØØØOOØ',
            'ØOOOOOOOOOØOOOOOOOOØ',
            'OOOOOOOOOOOOOOØØØOOO',
            'OØØØOOØØØOOOOOOOOOOO',
            'OOOOOOOOOOOOOOØØPOOO',
            'OØØØOOØØØOOOOOOOOOOØ',
            'OOOOOOOOOOOOOOOOOOOØ',
            'OØØØOOØØØOOOOOOØOOOØ',
            'OOOOOOOOOOOOOOOØOOOØ',
            'OØØØOOØØØOOOOOOØOOOØ',
            'OOOOOOOOOOOOOOOOOOOØ',
            'OØØØOOØØØOOOOOOOOOOØ',
            'OOOOOOOOOOØOOOØØOOOØ',
            'OØØØOOØØØOØOOOOOOOOØ',
            'OOPOOOOOOOØOOOOOOOOO',
            'OOOOOOOOOOØOOOOOOOOO',
            'ØØØØOOØØØØØOOOOØØPOO'
        ]
        self.size = 20
        self.warehouse = np.zeros((self.size, self.size), dtype=int)
        for r, row in enumerate(self.layout_str):
            for c, ch in enumerate(row):
                if ch == 'Ø': self.warehouse[r, c] = 1

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(17,), dtype=np.float32)

        self.agents = [None, None]
        self.prev_agents = [None, None]
        self.task_queue = []
        self.current_targets = [None, None]
        self.distance_maps = [None, None]

        # Timers
        self.task_timers = [0, 0]
        self.task_limit = 80

        # Stuck Counters (Individual)
        self.stuck_timers = [0, 0]

        # Global Task Counter (For "Numbering each task")
        self.global_task_count = 0

    def _get_random_free_cell(self):
        while True:
            r = np.random.randint(0, self.size)
            c = np.random.randint(0, self.size)
            if self.warehouse[r, c] == 0: return (r, c)

    def _update_distance_map(self, agent_idx):
        target = self.current_targets[agent_idx]
        if target is None: return
        dmap = np.full((self.size, self.size), -1, dtype=float)
        q = deque([(target, 0)])
        visited = set([target])
        dmap[target] = 0
        while q:
            (r, c), dist = q.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.warehouse[nr, nc] == 0 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        dmap[nr, nc] = dist + 1
                        q.append(((nr, nc), dist + 1))
        dmap[dmap == -1] = 999.0
        self.distance_maps[agent_idx] = dmap

    def reset(self):
        self.agents[0] = self._get_random_free_cell()
        while True:
            self.agents[1] = self._get_random_free_cell()
            if self.agents[1] != self.agents[0]: break
        self.prev_agents = list(self.agents)
        self.task_timers = [0, 0]
        self.global_task_count = 0
        self.generate_new_task_batch()
        return None

    def generate_new_task_batch(self):
        self.task_queue = []
        for _ in range(3):
            t = self._get_random_free_cell()
            self.task_queue.append(t)
        self.assign_targets()

    def assign_targets(self):
        for i in range(2):
            if self.current_targets[i] is None:
                if len(self.task_queue) > 0:
                    self.current_targets[i] = self.task_queue.pop(0)
                    self._update_distance_map(i)
                    self.task_timers[i] = 0

    def force_reset_task(self, agent_idx):
        print(f"!!! Timeout Robot {agent_idx + 1}: Skipping Task !!!")
        self.current_targets[agent_idx] = self._get_random_free_cell()
        self._update_distance_map(agent_idx)
        self.task_timers[agent_idx] = 0

    def get_agent_obs(self, agent_idx):
        me = self.agents[agent_idx]
        other = self.agents[1 - agent_idx]
        target = self.current_targets[agent_idx]
        prev = self.prev_agents[agent_idx]
        if target is None: target = me

        r, c = me
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        wall_dists = []
        for dr, dc in directions:
            dist = 0
            curr_r, curr_c = r, c
            while True:
                curr_r += dr
                curr_c += dc

                # Bounds
                if not (0 <= curr_r < self.size and 0 <= curr_c < self.size): break
                # Wall
                if self.warehouse[curr_r, curr_c] == 1: break
                # Other Robot (New Sensor Logic)
                if (curr_r, curr_c) == other: break

                dist += 1
            wall_dists.append(dist / self.size)

        if self.distance_maps[agent_idx] is not None:
            true_dist = self.distance_maps[agent_idx][r, c] / 50.0
        else:
            true_dist = 0.0

        obs_list = [
                       me[0] / self.size, me[1] / self.size,
                       target[0] / self.size, target[1] / self.size,
                       true_dist,
                       prev[0] / self.size, prev[1] / self.size,
                       other[0] / self.size, other[1] / self.size
                   ] + wall_dists
        return np.array(obs_list, dtype=np.float32)


def run_dual_robot_demo():
    env = DualRobotEnv()
    print("Loading Sensor-Aware Brain (v2)...")

    if not os.path.exists("ppo_multi_agent_v2.zip"):
        print("Error: Train 'ppo_multi_agent_v2' first!")
        return

    model = PPO.load("ppo_multi_agent_v2")
    env.reset()
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    paths = [[], []]

    for step in range(5000):
        # 1. AI Prediction
        actions = []
        for i in range(2):
            obs = env.get_agent_obs(i)
            action, _ = model.predict(obs, deterministic=True)
            actions.append(action)

        # --- FIX: INDIVIDUAL STUCK MANAGER ---
        for i in range(2):
            if env.agents[i] == env.prev_agents[i]:
                env.stuck_timers[i] += 1
            else:
                env.stuck_timers[i] = 0

            # If stuck for 3 frames, FORCE random move (The "Nudge")
            if env.stuck_timers[i] > 3:
                actions[i] = np.random.randint(0, 4)
                env.stuck_timers[i] = 0

        env.prev_agents = list(env.agents)

        # 2. Move Agents
        for i in range(2):
            if env.current_targets[i] is None: continue

            env.task_timers[i] += 1
            if env.task_timers[i] > env.task_limit:
                env.force_reset_task(i)
                continue

            r, c = env.agents[i]
            deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            dr, dc = deltas[actions[i]]
            nr, nc = r + dr, c + dc

            if 0 <= nr < env.size and 0 <= nc < env.size and env.warehouse[nr, nc] == 0:
                other_agent = env.agents[1 - i]
                if (nr, nc) != other_agent:
                    env.agents[i] = (nr, nc)

            # Completion Check
            if env.agents[i] == env.current_targets[i]:
                env.global_task_count += 1
                print(f"Task #{env.global_task_count} completed by Robot {i + 1}!")
                env.current_targets[i] = None
                env.assign_targets()

        if env.current_targets[0] is None and env.current_targets[1] is None and len(env.task_queue) == 0:
            print("--- BATCH COMPLETE ---")
            env.generate_new_task_batch()
            paths = [[], []]

            # Visuals
        for i in range(2): paths[i].append(env.agents[i])

        if step % 2 == 0:
            ax.clear()
            ax.imshow(env.warehouse, cmap='binary')
            r0, c0 = zip(*paths[0]) if len(paths[0]) > 0 else ([], [])
            r1, c1 = zip(*paths[1]) if len(paths[1]) > 0 else ([], [])
            ax.plot(c0, r0, color='magenta', alpha=0.6, linewidth=2)
            ax.plot(c1, r1, color='cyan', alpha=0.6, linewidth=2)
            ax.scatter(env.agents[0][1], env.agents[0][0], s=150, color='red', label='R1')
            ax.scatter(env.agents[1][1], env.agents[1][0], s=150, color='blue', label='R2')

            for i, t in enumerate(env.current_targets):
                if t:
                    color = 'red' if i == 0 else 'blue'
                    ax.scatter(t[1], t[0], s=100, marker='x', color=color)

            for q in env.task_queue:
                ax.scatter(q[1], q[0], s=50, marker='x', color='grey', alpha=0.5)

            ax.legend(loc='upper right')
            ax.set_title(f"Total Completed Tasks: {env.global_task_count}")
            plt.pause(0.1)


if __name__ == "__main__":
    run_dual_robot_demo()