import numpy as np
import matplotlib.pyplot as plt

from typing_env import TypingEnv
from agents.q_learning import QLearningAgent
from agents.dqn_agent import DQNAgent

from agents.reinforce import ReinforceAgent
from agents.actor_critic import ActorCriticAgent
from agents.ppo import PPOAgent

import random
import torch

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# -------------------------------
# Rule-based agent
# -------------------------------
def select_action(env):
    # scores = env.k - 0.1 * env.t
    scores = env.k
    b = np.argmin(scores)
    
    skill = env.k[b]
    
    if skill < 0.4:
        d = 0
    elif skill < 0.6:
        d = 1
    elif skill < 0.75:
        d = 2
    elif skill < 0.9:
        d = 3
    else:
        d = 4
        
    return b * env.L + d


# -------------------------------
# Rule-based runner
# -------------------------------
def run_rule(env, episodes=300, steps=300):
    rewards = []
    avg_skills = []
    min_skills = []
    std_skills = []

    for _ in range(episodes):
        env.reset()
        ep_rewards = []
        
        for _ in range(steps):
            action = select_action(env)
            _, r, _, _ = env.step(action)
            ep_rewards.append(r)
        
        rewards.append(np.mean(ep_rewards))
        avg_skills.append(np.mean(env.k))
        min_skills.append(np.min(env.k))
        std_skills.append(np.std(env.k))
        
    return rewards, avg_skills, min_skills, std_skills


# -------------------------------
# Main comparison
# -------------------------------
def compare():
    results = []
    labels = []

    # ---- Rule ----
    print("Running Rule...")
    env_rule = TypingEnv()
    res = run_rule(env_rule)
    results.append(res)
    labels.append("Rule")

    # ---- Q-Learning ----
    print("Running Q-learning...")
    q_agent = QLearningAgent()
    res = q_agent.train(episodes=300)
    results.append(res)
    labels.append("Q-Learning")

    # ---- DQN ----
    print("Running DQN...")
    dqn_agent = DQNAgent()
    res = dqn_agent.train(episodes=300)
    results.append(res)
    labels.append("DQN")

    # ---- REINFORCE ----
    print("Running REINFORCE...")
    rf_agent = ReinforceAgent()
    res = rf_agent.train(episodes=300)
    results.append(res)
    labels.append("REINFORCE")

    # ---- Actor-Critic ----
    print("Running Actor-Critic...")
    ac_agent = ActorCriticAgent()
    res = ac_agent.train(episodes=300)
    results.append(res)
    labels.append("Actor-Critic")

    # ---- PPO ----
    print("Running PPO...")
    ppo_env = TypingEnv()
    ppo_agent = PPOAgent(ppo_env)
    res = ppo_agent.train(episodes=300)
    results.append(res)
    labels.append("PPO")

    return results, labels

def smooth(x, window=10):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode='same')


def print_summary(results, labels):
    print("\n===== FINAL METRICS =====")
    for (r, avg, min_s, std_s), label in zip(results, labels):
        print(f"{label}:")
        print(f"  Final Avg Skill: {avg[-1]:.4f}")
        print(f"  Final Min Skill: {min_s[-1]:.4f}")
        print(f"  Final Std Dev  : {std_s[-1]:.4f}")
        print("-" * 40)



# -------------------------------
# Plotting
# -------------------------------
def plot_all(results, labels):
    plt.figure(figsize=(16, 10))

    # ---- Reward ----
    plt.subplot(2, 2, 1)
    for (r, _, _, _), label in zip(results, labels):
        plt.plot(smooth(r), label=label)
    plt.title("Reward")
    plt.legend()

    # ---- Avg Skill ----
    plt.subplot(2, 2, 2)
    for (_, avg, _, _), label in zip(results, labels):
        plt.plot(smooth(avg), label=label)
    plt.title("Average Skill")
    plt.legend()

    # ---- Min Skill ----
    plt.subplot(2, 2, 3)
    for (_, _, min_s, _), label in zip(results, labels):
        plt.plot(smooth(min_s), label=label)
    plt.title("Minimum Skill (Weakest Bigram)")
    plt.legend()

    # ---- Std ----
    plt.subplot(2, 2, 4)
    for (_, _, _, std_s), label in zip(results, labels):
        plt.plot(smooth(std_s), label=label)
    plt.title("Skill Std Deviation")
    plt.legend()

    plt.tight_layout()
    plt.savefig("figs/compare_all_agents.png")
    plt.show()


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    results, labels = compare()
    print_summary(results, labels)
    plot_all(results, labels)
