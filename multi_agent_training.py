import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from collections import deque

class MultiAgentTrainingEnv(gym.Env):
    def __init__(self):
        super(MultiAgentTrainingEnv, self).__init__()

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

        # 17 Inputs: [Agent(2), Target(2), Dist(1), Memory(2), 8-Lidar(8), Other_Agent(2)]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(17,), dtype=np.float32)

        self.agent_pos = None
        self.prev_agent_pos = None
        self.target_pos = None
        self.other_agent_pos = None

        self.max_steps = 1000
        self.steps_taken = 0
        self.distance_map = None

    def _get_random_free_cell(self):
        while True:
            r = np.random.randint(0, self.size)
            c = np.random.randint(0, self.size)
            if self.warehouse[r, c] == 0: return (r, c)

    def _update_distance_map(self):
        self.distance_map = np.full((self.size, self.size), -1, dtype=float)
        q = deque([(self.target_pos, 0)])
        visited = set([self.target_pos])
        self.distance_map[self.target_pos] = 0

        while q:
            (r, c), dist = q.popleft()
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.warehouse[nr, nc] == 0 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        self.distance_map[nr, nc] = dist + 1
                        q.append(((nr, nc), dist + 1))
        self.distance_map[self.distance_map == -1] = 999.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self._get_random_free_cell()
        self.prev_agent_pos = self.agent_pos

        while True:
            self.other_agent_pos = self._get_random_free_cell()
            if self.other_agent_pos != self.agent_pos: break

        while True:
            self.target_pos = self._get_random_free_cell()
            if self.target_pos != self.agent_pos and self.target_pos != self.other_agent_pos: break

        self._update_distance_map()
        self.steps_taken = 0
        return self._get_obs(), {}

    def _get_obs(self):
        r, c = self.agent_pos
        pr, pc = self.prev_agent_pos
        tr, tc = self.target_pos
        or_r, or_c = self.other_agent_pos

        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        wall_dists = []
        for dr, dc in directions:
            dist = 0
            curr_r, curr_c = r, c
            while True:
                curr_r += dr
                curr_c += dc

                # === NEW SENSOR LOGIC ===
                # Check bounds
                if not (0 <= curr_r < self.size and 0 <= curr_c < self.size):
                    break

                # Check Wall
                if self.warehouse[curr_r, curr_c] == 1:
                    break

                # Check Other Agent (Treat as wall)
                if (curr_r, curr_c) == (or_r, or_c):
                    break

                dist += 1

            wall_dists.append(dist / self.size)

        true_dist = self.distance_map[r, c] / 50.0

        obs_list = [
            r / self.size, c / self.size,
            tr / self.size, tc / self.size,
            true_dist,
            pr / self.size, pc / self.size,
            or_r / self.size, or_c / self.size
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

        if 0 <= nr < self.size and 0 <= nc < self.size and self.warehouse[nr, nc] == 0:
            self.agent_pos = (nr, nc)
        else:
            hit_wall = True
            reward -= 5.0

        # Ghost moves randomly
        gr, gc = self.other_agent_pos
        g_action = np.random.randint(0, 4)
        g_dr, g_dc = deltas[g_action]
        gnr, gnc = gr + g_dr, gc + g_dc
        if 0 <= gnr < self.size and 0 <= gnc < self.size and self.warehouse[gnr, gnc] == 0:
            if (gnr, gnc) != self.agent_pos:
                self.other_agent_pos = (gnr, gnc)

        if self.agent_pos == self.other_agent_pos:
            reward -= 10.0
            self.agent_pos = self.prev_agent_pos

        new_dist = self.distance_map[self.agent_pos]

        if not hit_wall:
            reward += (old_dist - new_dist) * 0.5

            # === INCREASED IDLE PENALTY ===
            # Prevents sticking next to walls.
            # It is now more painful to sit still (-1.0) than to move away from target (-0.5)
            if self.agent_pos == (r, c):
                reward -= 1.0

        if self.agent_pos == self.target_pos:
            reward += 100.0
            terminated = True

        if self.steps_taken >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

if __name__ == '__main__':
    env = MultiAgentTrainingEnv()
    env = Monitor(env)

    print("Training Sensor-Aware Model (v2)...")

    policy_kwargs = dict(net_arch=[256, 256])

    model = PPO("MlpPolicy", env, verbose=1,
                learning_rate=0.0003,
                ent_coef=0.005,
                gamma=0.85,
                policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=2000000)

    model.save("ppo_multi_agent_v2")
    print("Saved as 'ppo_multi_agent_v2.zip'")