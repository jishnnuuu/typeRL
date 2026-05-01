import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np

from typing_env import TypingEnv
from agents.dqn_agent import DQNAgent
from agents.ppo import PPOAgent

from forgetting_configs import FORGETTING_CONFIGS


def run_experiment(agent_type, forget_type, episodes=200):
    print(f"\nRunning {agent_type} | {forget_type}")

    env = TypingEnv()
    env.forgetting_type = forget_type

    if agent_type == "DQN":
        agent = DQNAgent()
        agent.env.forgetting_type = forget_type
        rewards, avg, min_s, std = agent.train(episodes=episodes)

    elif agent_type == "PPO":
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
        for name, ftype in FORGETTING_CONFIGS.items():
            res = run_experiment(agent, ftype)
            results[agent][name] = res

    os.makedirs("ablation/results", exist_ok=True)

    with open("ablation/results/forgetting_ablation.pkl", "wb") as f:
        pickle.dump(results, f)

    print("\nSaved results!")


if __name__ == "__main__":
    run_all()