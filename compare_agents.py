import numpy as np
import matplotlib.pyplot as plt

from typing_env import TypingEnv
from q_learning import QLearningAgent
from dqn_agent import DQNAgent

# random agent
def run_random(env, episodes=50, steps=200):
    rewards, skills = [], []
    
    for _ in range(episodes):
        env.reset()
        ep_rewards = []
        for _ in range(steps):
            action = np.random.randint(env.K * env.L)
            _, r, _, _ = env.step(action)
            ep_rewards.append(r)
        rewards.append(np.mean(ep_rewards))
        skills.append(np.mean(env.k))
    return rewards, skills


# rule based agent
def select_action(env):
    scores = env.k - 0.1 * env.t
    b = np.argmin(scores)
    
    skill = env.k[b]
    
    if skill < 0.3:
        d = 0
    elif skill < 0.5:
        d = 1
    elif skill < 0.7:
        d = 2
    elif skill < 0.85:
        d = 3
    else:
        d = 4
    return b * env.L + d


def run_rule(env, episodes=50, steps=200):
    rewards, skills = [], []
    for _ in range(episodes):
        env.reset()
        ep_rewards = []
        
        for _ in range(steps):
            action = select_action(env)
            _, r, _, _ = env.step(action)
            ep_rewards.append(r)
            
        rewards.append(np.mean(ep_rewards))
        skills.append(np.mean(env.k))
    return rewards, skills


# main comparision
def compare():
    env = TypingEnv()
    
    print("Running Random...")
    r_r, s_r = run_random(env)
    
    print("Running Rule...")
    r_rb, s_rb = run_rule(env)
    
    print("Running Q-learning...")
    q_agent = QLearningAgent()
    r_q, s_q = q_agent.train(episodes=50)
    
    print("Running DQN...")
    dqn_agent = DQNAgent()
    r_dqn, s_dqn = dqn_agent.train(episodes=50)
    
    return (r_r, s_r), (r_rb, s_rb), (r_q, s_q), (r_dqn, s_dqn)


def plot_all(results):
    labels = ["Random", "Rule", "Q-Learning", "DQN"]
    plt.figure(figsize=(12,5))
    
    # Reward
    plt.subplot(1,2,1)
    for (r, _), label in zip(results, labels):
        plt.plot(r, label=label)
    plt.title("Reward Comparison")
    plt.legend()
    
    # Skill
    plt.subplot(1,2,2)
    for (_, s), label in zip(results, labels):
        plt.plot(s, label=label)
    plt.title("Skill Comparison")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("figs/compare_agents.png")
    plt.show()


if __name__ == "__main__":
    results = compare()
    plot_all(results)