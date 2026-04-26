Warehouse Multi-Agent Reinforcement Learning

Introduction
This repository contains the source code developed for the Individual Project: "Scalable Multi-Agent Path Planning in Dynamic Warehouse Environments Using Reinforcement Learning" at the University of Manchester (2025/26).

The software implements a decentralised Multi-Agent Proximal Policy Optimisation (MAPPO) framework for autonomous robot navigation in a simulated warehouse environment. Robots learn to navigate a 20x20 grid-world warehouse, avoiding static obstacles and dynamic inter-agent collisions whilst completing transportation tasks.

Repository Structure
- `multi_agent_training.py` — Single-agent training environment with a random 
  ghost agent. Run this first to produce the base policy (ppo_multi_agent_v2)
- `multi_agent_sim.py` — Dual robot simulation using the trained base policy
- `multiple_robots.py` — Scalable N-robot training environment with greedy 
  ghost agents. Produces the scalable policy
- `multiple_agent_sim.py` — N-robot simulation demo using the scalable policy

Installation
The following libraries are required:
pip install numpy
pip install gymnasium
pip install stable-baselines3
pip install matplotlib

How to Run
1. Train the base single-agent policy
2. Run the dual robot simulation
3. Train the scalable multi-agent policy
4. Run the N-robot simulation (the number of robots can be adjusted by changing the num_robots 'parameter'

Technical details
1. Environment: Custom Gymnasium warehouse grid (20x20)
2. Algorithm: Proximal Policy Optimisation (PPO) via Stable-Baselines3
3. Observation space: 17-dimensional vector (agent position, target position, BFS distance, positional memory and 8-direction Lidar)
4. Action space: Discrete (4 actions - up, down, left, right)
5. Reward Shaping: BFS-guided dense reward function

Known Issues:
- Training times for the scalable model can be significant. Timesteps may need to be reduced
- Simulation visualisation may run slower on certain devices
- LLM assistance was used to accelerate script development in this project

Dependencies
- numpy
- gymnasium >= 0.29
- stable-baselines3
- matplotlib
