""" 
RULE BASED AGENT
    - Greedy Curriculum
1. Find weakest bigram (lowest k[b])
2. Choose difficulty based on skill
3. Train that bigram
"""

import numpy as np
import matplotlib.pyplot as plt

from typing_env import TypingEnv


def select_action(env):
    # 1. find weakest bigram
    weakest_bigram = np.argmin(env.k)
    
    # 2. choose difficulty based on skill
    skill = env.k[weakest_bigram]
    
    if skill < 0.3:
        difficulty = 0
    elif skill < 0.5:
        difficulty = 1
    elif skill < 0.7:
        difficulty = 2
    elif skill < 0.85:
        difficulty = 3
    else:
        difficulty = 4
    # encode action
    action = weakest_bigram * env.L + difficulty
    return action

def run_rule_agent(episodes=30, steps_per_episode=100):
    env = TypingEnv()
    
    all_rewards = []
    all_skills = []
    
    tracked_bigram = 0
    tracked_skill = []
    
    for ep in range(episodes):
        state = env.reset()
        
        episode_rewards = []
        episode_skills = []
        
        tracked_skill.append(env.k[tracked_bigram])
        
        for step in range(steps_per_episode):
            action = select_action(env)
            state, reward, done, _ = env.step(action)
            
            episode_rewards.append(reward)
            episode_skills.append(np.mean(env.k))
            
        all_rewards.append(np.mean(episode_rewards))
        all_skills.append(episode_skills[-1])
        
        print(f"Episode {ep+1}: Skill = {all_skills[-1]:.4f}")
    return all_rewards, all_skills, tracked_skill

def plot_results(rewards, skills, tracked):
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,3,1)
    plt.plot(rewards)
    plt.title("Reward")
    
    plt.subplot(1,3,2)
    plt.plot(skills)
    plt.title("Final Skill")
    
    plt.subplot(1,3,3)
    plt.plot(tracked)
    plt.title("Tracked Bigram")
    
    plt.tight_layout()
    plt.savefig("figs/rule_based_agent.png")
    plt.show()
    
if __name__ == "__main__":
    rewards, skills, tracked = run_rule_agent()
    plot_results(rewards, skills, tracked)