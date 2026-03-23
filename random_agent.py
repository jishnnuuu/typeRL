""" 
RANDOM AGENT
“How does the system behave WITHOUT intelligence?”

at each step:
action = random integer in [0, K × L)

No learning, no policy
"""

import numpy as np
import matplotlib.pyplot as plt

from typing_env import TypingEnv


def run_random_agent(episodes=30, steps_per_episode=100):
    env = TypingEnv()
    
    K = env.K
    L = env.L
    
    all_rewards = []
    all_skills = []
    
    tracked_bigram = 0
    tracked_skill = []
    
    for ep in range(episodes):
        state = env.reset()
        
        episode_rewards = []
        episode_skills = [] 
        
        for step in range(steps_per_episode):
            # random action
            action = np.random.randint(0, K * L)
            
            state, reward, done, _ = env.step(action)
            
            tracked_skill.append(env.k[tracked_bigram])
            
            episode_rewards.append(reward)
            episode_skills.append(np.mean(env.k))
        all_rewards.append(np.mean(episode_rewards))
        # this shows final skills achieved in that episode
        all_skills.append(episode_skills[-1])
        print(f"Episode {ep+1}: Avg Reward = {all_rewards[-1]:.4f}, Avg Skill = {all_skills[-1]:.4f}")
    return all_rewards, all_skills, tracked_skill

def plot_results(rewards, skills, tracked_skill):
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,3,1)
    plt.plot(rewards)
    plt.title("Avg Reward")
    
    plt.subplot(1,3,2)
    plt.plot(skills)
    plt.title("Final Skill per Episode")
    
    plt.subplot(1,3,3)
    plt.plot(tracked_skill)
    plt.title("Tracked Bigram Skill")
    
    plt.tight_layout()
    plt.savefig("figs/random_agent.png")
    plt.show()
    
if __name__ == "__main__":
    rewards, skills,tracked_skill  = run_random_agent()
    plot_results(rewards, skills, tracked_skill)