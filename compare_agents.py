import numpy as np
import matplotlib.pyplot as plt

from typing_env import TypingEnv
from agents.q_learning import QLearningAgent
from agents.dqn_agent import DQNAgent


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
    print("Running Rule...")
    env_rule = TypingEnv()
    r_rb, avg_rb, min_rb, std_rb = run_rule(env_rule)

    print("Running Q-learning...")
    q_agent = QLearningAgent()
    r_q, avg_q, min_q, std_q = q_agent.train(episodes=300)

    print("Running DQN...")
    dqn_agent = DQNAgent()
    r_dqn, avg_dqn, min_dqn, std_dqn = dqn_agent.train(episodes=300)

    return (
        (r_rb, avg_rb, min_rb, std_rb),
        (r_q, avg_q, min_q, std_q),
        (r_dqn, avg_dqn, min_dqn, std_dqn),
    )


# -------------------------------
# Plotting
# -------------------------------
def plot_all(results):
    labels = ["Rule", "Q-Learning", "DQN"]

    plt.figure(figsize=(15, 8))

    # ---- Reward ----
    plt.subplot(2, 2, 1)
    for (r, _, _, _), label in zip(results, labels):
        plt.plot(r, label=label)
    plt.title("Reward")
    plt.legend()

    # ---- Average Skill ----
    plt.subplot(2, 2, 2)
    for (_, avg, _, _), label in zip(results, labels):
        plt.plot(avg, label=label)
    plt.title("Average Skill")
    plt.legend()

    # ---- Minimum Skill ----
    plt.subplot(2, 2, 3)
    for (_, _, min_s, _), label in zip(results, labels):
        plt.plot(min_s, label=label)
    plt.title("Minimum Skill (Weakest Bigram)")
    plt.legend()

    # ---- Variance ----
    plt.subplot(2, 2, 4)
    for (_, _, _, std_s), label in zip(results, labels):
        plt.plot(std_s, label=label)
    plt.title("Skill Variance")
    plt.legend()

    plt.tight_layout()
    plt.savefig("figs/compare_agents_detailed.png")
    plt.show()


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    results = compare()
    plot_all(results)
