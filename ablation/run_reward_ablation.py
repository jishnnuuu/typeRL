import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle

from typing_env import TypingEnv
from agents.dqn_agent import DQNAgent
from agents.ppo import PPOAgent

from reward_configs import REWARD_CONFIGS


def run_experiment(agent_type, config_name, reward_weights, episodes=200):
    print(f"\nRunning {agent_type} | {config_name}")

    if agent_type == "DQN":
        agent = DQNAgent()
        agent.env.reward_weights = reward_weights
        rewards, avg, min_s, std = agent.train(episodes=episodes)

    elif agent_type == "PPO":
        env = TypingEnv()
        env.reward_weights = reward_weights
        agent = PPOAgent(env)
        rewards, avg, min_s, std = agent.train(episodes=episodes)
    else:
        raise ValueError("Unknown agent")

    return {
        "reward": rewards,
        "avg": avg,
        "min": min_s,
        "std": std,
    }


def run_all():
    results = {
        "DQN": {},
        "PPO": {}
    }

    for agent in ["DQN", "PPO"]:
        for name, config in REWARD_CONFIGS.items():
            res = run_experiment(agent, name, config)
            results[agent][name] = res

    os.makedirs("ablation/results", exist_ok=True)

    with open("ablation/results/reward_ablation.pkl", "wb") as f:
        pickle.dump(results, f)

    print("\nSaved results!")


if __name__ == "__main__":
    run_all()