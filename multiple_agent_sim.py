import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from collections import deque
import os
import random


class MultiRobotEnv(gym.Env):
    def __init__(self, num_robots=5):
        super(MultiRobotEnv, self).__init__()
        self.num_robots = num_robots

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

        # Dynamic lists based on number of robots
        self.agents = [None] * self.num_robots
        self.prev_agents = [None] * self.num_robots
        self.current_targets = [None] * self.num_robots
        self.distance_maps = [None] * self.num_robots

        # Timers
        self.task_timers = [0] * self.num_robots
        self.task_limit = 80
        self.stuck_timers = [0] * self.num_robots

        self.task_queue = []
        self.global_task_count = 0

    def _get_random_free_cell(self, exclude_positions=[]):
        while True:
            r = np.random.randint(0, self.size)
            c = np.random.randint(0, self.size)
            if self.warehouse[r, c] == 0 and (r, c) not in exclude_positions:
                return (r, c)

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
        occupied = []
        for i in range(self.num_robots):
            pos = self._get_random_free_cell(exclude_positions=occupied)
            self.agents[i] = pos
            occupied.append(pos)

        self.prev_agents = list(self.agents)
        self.task_timers = [0] * self.num_robots
        self.stuck_timers = [0] * self.num_robots
        self.global_task_count = 0
        self.generate_new_task_batch()
        return None

    def generate_new_task_batch(self):
        self.task_queue = []
        # Generate twice as many tasks as there are robots
        for _ in range(self.num_robots * 2):
            t = self._get_random_free_cell()
            self.task_queue.append(t)
        self.assign_targets()

    def assign_targets(self):
        for i in range(self.num_robots):
            if self.current_targets[i] is None:
                if len(self.task_queue) > 0:
                    self.current_targets[i] = self.task_queue.pop(0)
                    self._update_distance_map(i)
                    self.task_timers[i] = 0

    def force_reset_task(self, agent_idx):
        print(f"!!! Timeout Robot {agent_idx + 1}: Skipping Task !!!")
        self.current_targets[agent_idx] = self._get_random_free_cell(exclude_positions=self.agents)
        self._update_distance_map(agent_idx)
        self.task_timers[agent_idx] = 0

    def get_agent_obs(self, agent_idx):
        me = self.agents[agent_idx]
        target = self.current_targets[agent_idx]
        prev = self.prev_agents[agent_idx]
        if target is None: target = me

        # Get all other agents' positions
        other_positions = set(self.agents) - {me}

        r, c = me
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        wall_dists = []
        for dr, dc in directions:
            dist = 0
            curr_r, curr_c = r, c
            while True:
                curr_r += dr
                curr_c += dc
                if not (0 <= curr_r < self.size and 0 <= curr_c < self.size): break
                if self.warehouse[curr_r, curr_c] == 1: break
                if (curr_r, curr_c) in other_positions: break
                dist += 1
            wall_dists.append(dist / self.size)

        if self.distance_maps[agent_idx] is not None:
            true_dist = self.distance_maps[agent_idx][r, c] / 50.0
        else:
            true_dist = 0.0

        # Find the closest other agent (for the last 2 inputs of the observation)
        closest_r, closest_c = me
        min_dist = 999
        for op in other_positions:
            dist_to_op = abs(op[0] - r) + abs(op[1] - c)
            if dist_to_op < min_dist:
                min_dist = dist_to_op
                closest_r, closest_c = op

        obs_list = [
                       me[0] / self.size, me[1] / self.size,
                       target[0] / self.size, target[1] / self.size,
                       true_dist,
                       prev[0] / self.size, prev[1] / self.size,
                       closest_r / self.size, closest_c / self.size
                   ] + wall_dists

        return np.array(obs_list, dtype=np.float32)


def run_multi_robot_demo(num_robots=5):
    env = MultiRobotEnv(num_robots=num_robots)
    print("Loading Sensor-Aware Brain...")

    if not os.path.exists("test_model_v3.zip"):
        print("Error: Train 'test_model_v3.zip' first!")
        return

    model = PPO.load("test_model_v3.zip")
    env.reset()
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    # Store path histories for trails
    paths = [[] for _ in range(num_robots)]

    # Generate distinct colors for each robot
    colors = cm.rainbow(np.linspace(0, 1, num_robots))

    for step in range(5000):
        actions = []

        # 1. AI Prediction for EVERY robot independently
        for i in range(num_robots):
            obs = env.get_agent_obs(i)
            action, _ = model.predict(obs, deterministic=True)
            actions.append(action)

        # 2. Individual Stuck Manager
        for i in range(num_robots):
            if env.agents[i] == env.prev_agents[i]:
                env.stuck_timers[i] += 1
            else:
                env.stuck_timers[i] = 0

            # If stuck for 3 frames, FORCE random move (The "Nudge")
            if env.stuck_timers[i] > 3:
                actions[i] = np.random.randint(0, 4)
                env.stuck_timers[i] = 0

        env.prev_agents = list(env.agents)

        # 3. Move Agents
        for i in range(num_robots):
            if env.current_targets[i] is None: continue

            env.task_timers[i] += 1
            if env.task_timers[i] > env.task_limit:
                env.force_reset_task(i)
                continue

            r, c = env.agents[i]
            deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            dr, dc = deltas[actions[i]]
            nr, nc = r + dr, c + dc

            # Collision Check against walls AND all other agents
            if 0 <= nr < env.size and 0 <= nc < env.size and env.warehouse[nr, nc] == 0:
                if (nr, nc) not in env.agents:
                    env.agents[i] = (nr, nc)

            # Completion Check
            if env.agents[i] == env.current_targets[i]:
                env.global_task_count += 1
                print(f"Task #{env.global_task_count} completed by Robot {i + 1}!")
                env.current_targets[i] = None
                env.assign_targets()

        # Check if batch is completely done
        if all(t is None for t in env.current_targets) and len(env.task_queue) == 0:
            print("--- BATCH COMPLETE ---")
            env.generate_new_task_batch()
            paths = [[] for _ in range(num_robots)]

        # 4. Visuals
        for i in range(num_robots):
            paths[i].append(env.agents[i])
            # Keep trail from getting infinitely long
            if len(paths[i]) > 30:
                paths[i].pop(0)

        if step % 2 == 0:
            ax.clear()
            ax.imshow(env.warehouse, cmap='binary')

            for i in range(num_robots):
                r_coords, c_coords = zip(*paths[i]) if len(paths[i]) > 0 else ([], [])
                # Draw path trail
                ax.plot(c_coords, r_coords, color=colors[i], alpha=0.4, linewidth=2)
                # Draw Robot
                ax.scatter(env.agents[i][1], env.agents[i][0], s=150, color=colors[i], label=f'R{i + 1}')

                # Draw assigned target
                t = env.current_targets[i]
                if t:
                    ax.scatter(t[1], t[0], s=100, marker='x', color=colors[i], linewidths=3)

            # Draw unassigned tasks in the queue
            for q in env.task_queue:
                ax.scatter(q[1], q[0], s=50, marker='x', color='grey', alpha=0.5)

            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
            ax.set_title(f"Active Robots: {num_robots} | Total Completed Tasks: {env.global_task_count}")
            plt.pause(0.1)


if __name__ == "__main__":
    # Change this number to test scalability! (e.g., 5, 10, 15)
    run_multi_robot_demo(num_robots=5)