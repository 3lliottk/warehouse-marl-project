import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results
from collections import deque
import os
import matplotlib.pyplot as plt


# ==========================================
# 1. THE ENVIRONMENT
# ==========================================
class MultiAgentTrainingEnv(gym.Env):
    def __init__(self, num_robots=2):
        super(MultiAgentTrainingEnv, self).__init__()

        self.num_robots = num_robots
        self.num_other_agents = num_robots - 1

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

        self.action_space = spaces.Discrete(4)

        # 17 Inputs: [Agent(2), Target(2), Dist(1), Memory(2), 8-Lidar(8), Closest_Other_Agent(2)]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(17,), dtype=np.float32)

        self.agent_pos = None
        self.prev_agent_pos = None
        self.target_pos = None

        # List to hold state of all other robots
        self.other_agents = []

        self.max_steps = 1000
        self.steps_taken = 0
        self.distance_map = None

    def _get_random_free_cell(self, exclude_positions=[]):
        while True:
            r = np.random.randint(0, self.size)
            c = np.random.randint(0, self.size)
            if self.warehouse[r, c] == 0 and (r, c) not in exclude_positions:
                return (r, c)

    def _update_distance_map(self):
        self.distance_map = np.full((self.size, self.size), -1, dtype=float)
        q = deque([(self.target_pos, 0)])
        visited = set([self.target_pos])
        self.distance_map[self.target_pos] = 0

        while q:
            (r, c), dist = q.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.warehouse[nr, nc] == 0 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        self.distance_map[nr, nc] = dist + 1
                        q.append(((nr, nc), dist + 1))
        self.distance_map[self.distance_map == -1] = 999.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        occupied = []

        # 1. Spawn Ego Agent
        self.agent_pos = self._get_random_free_cell()
        self.prev_agent_pos = self.agent_pos
        occupied.append(self.agent_pos)

        # 2. Spawn Ego Target
        self.target_pos = self._get_random_free_cell(exclude_positions=occupied)
        occupied.append(self.target_pos)

        # 3. Spawn Other Agents and their targets
        self.other_agents = []
        for _ in range(self.num_other_agents):
            pos = self._get_random_free_cell(exclude_positions=occupied)
            occupied.append(pos)

            target = self._get_random_free_cell(exclude_positions=occupied)
            # We don't append targets to occupied so agents can share drop-off zones

            self.other_agents.append({'pos': pos, 'target': target})

        self._update_distance_map()
        self.steps_taken = 0
        return self._get_obs(), {}

    def _get_obs(self):
        r, c = self.agent_pos
        pr, pc = self.prev_agent_pos
        tr, tc = self.target_pos

        # Extract a set of other agent positions for Lidar to detect
        other_positions = set(agent['pos'] for agent in self.other_agents)

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

        true_dist = self.distance_map[r, c] / 50.0

        # Find the closest other agent for the final 2 observation slots
        closest_r, closest_c = r, c
        min_dist = 999
        for pos in other_positions:
            dist = abs(pos[0] - r) + abs(pos[1] - c)  # Manhattan distance
            if dist < min_dist:
                min_dist = dist
                closest_r, closest_c = pos[0], pos[1]

        obs_list = [
                       r / self.size, c / self.size,
                       tr / self.size, tc / self.size,
                       true_dist,
                       pr / self.size, pc / self.size,
                       closest_r / self.size, closest_c / self.size
                   ] + wall_dists

        return np.array(obs_list, dtype=np.float32)

    def step(self, action):
        self.steps_taken += 1
        self.prev_agent_pos = self.agent_pos

        r, c = self.agent_pos
        old_dist = self.distance_map[r, c]
        deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        dr, dc = deltas[action]
        nr, nc = r + dr, c + dc

        reward = -0.05
        terminated = False
        truncated = False
        hit_wall = False

        # Ego Agent Movement
        if 0 <= nr < self.size and 0 <= nc < self.size and self.warehouse[nr, nc] == 0:
            self.agent_pos = (nr, nc)
        else:
            hit_wall = True
            reward -= 5.0

        # Other Agents Movement (Greedy towards their specific targets)
        for idx, ghost in enumerate(self.other_agents):
            gr, gc = ghost['pos']
            gtr, gtc = ghost['target']

            possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            np.random.shuffle(possible_moves)  # Add slight stochasticity

            best_move = None
            best_dist = abs(gr - gtr) + abs(gc - gtc)  # Current distance

            for g_dr, g_dc in possible_moves:
                gnr, gnc = gr + g_dr, gc + g_dc
                if 0 <= gnr < self.size and 0 <= gnc < self.size and self.warehouse[gnr, gnc] == 0:
                    dist = abs(gnr - gtr) + abs(gnc - gtc)
                    if dist < best_dist:
                        best_dist = dist
                        best_move = (gnr, gnc)

            # Move the agent if a better move exists and it doesn't crash into the ego agent
            if best_move and best_move != self.agent_pos:
                self.other_agents[idx]['pos'] = best_move

            # Task Completion: Assign new target if reached
            if self.other_agents[idx]['pos'] == ghost['target']:
                self.other_agents[idx]['target'] = self._get_random_free_cell()

        # Inter-agent crash penalty (check against all other agents)
        for ghost in self.other_agents:
            if self.agent_pos == ghost['pos']:
                reward -= 25.0
                self.agent_pos = self.prev_agent_pos
                break  # Only penalize once per step

        new_dist = self.distance_map[self.agent_pos]

        # Reward Shaping
        if not hit_wall:
            reward += (old_dist - new_dist) * 0.5
            if self.agent_pos == (r, c):
                reward -= 1.0

        # Goal Achievement
        if self.agent_pos == self.target_pos:
            reward += 100.0
            terminated = True

        if self.steps_taken >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}



# 2. PLOTTING & TRAINING
def plot_clean_curve(log_folder, title='Training Progress (Smoothed)'):
    try:
        df = load_results(log_folder)
    except Exception as e:
        print(f"Error loading results from {log_folder}. Make sure training ran properly. Details: {e}")
        return

    if len(df) == 0:
        print("No data found in monitor.csv to plot.")
        return

    episodes = df.index.values
    rewards = df['r'].values

    window = 50
    if len(rewards) >= window:
        kernel = np.ones(window) / window
        y_smooth = np.convolve(rewards, kernel, mode='valid')
        x_smooth = episodes[window - 1:]
    else:
        y_smooth = rewards
        x_smooth = episodes

    plt.figure(figsize=(10, 6))
    plt.plot(x_smooth, y_smooth, color='#1f77b4', linewidth=2.5, label=f'Avg Reward ({window} ep moving avg)')
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Cumulative Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 1. Log Setup
    log_dir = "tmp_log/"
    os.makedirs(log_dir, exist_ok=True)

    # 2. Init Env with Monitor (Adjust num_robots here)
    env = MultiAgentTrainingEnv(num_robots=10)
    env = Monitor(env, log_dir)

    print("Training Scalable Multi-Agent Model...")

    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO("MlpPolicy", env, verbose=1,
                learning_rate=0.0003,
                ent_coef=0.02,
                gamma=0.60,
                policy_kwargs=policy_kwargs)

    # 3. Train
    try:
        model.learn(total_timesteps=5000000)
    except KeyboardInterrupt:
        print("\nTraining interrupted manually. Saving model...")

    model.save("scalable_policy")
    print("Saved as 'scalable_policy.zip'")

    # 4. Plot
    print("Generating Clean Learning Curve...")
    plot_clean_curve(log_dir)
