import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
import matplotlib.pyplot as plt
# Assuming typing_env exists in your local directory
from typing_env import TypingEnv

class QLearningAgent:
    def __init__(
        self,
        n_bins=50,
        alpha=0.3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.05,
    ):
        self.env = TypingEnv()
        
        self.K = self.env.K
        self.L = self.env.L
        self.n_actions = self.K * self.L
        
        # discretization
        self.n_bins = n_bins
        self.bins = np.linspace(0, 1, n_bins + 1)
        
        # Q-table: (state_space, action_space)
        self.Q = np.zeros((n_bins, self.n_actions))
        
        # learning params
        self.alpha = alpha
        self.gamma = gamma
        
        # exploration
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
    
    def discretize_state(self, k_vector):
        avg_skill = np.mean(k_vector)
        # digitize returns 1-based index, subtract 1 for 0-based
        bin_idx = np.digitize(avg_skill, self.bins) - 1
        # Clamp value to ensure it fits in the table
        return int(np.clip(bin_idx, 0, self.n_bins - 1))

    def select_action(self, state_bin):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state_bin])
    
    def update(self, s, a, r, s_next):
        best_next = np.max(self.Q[s_next])
        # Q-learning formula: Q(s,a) = Q(s,a) + alpha * (r + gamma * maxQ(s',a') - Q(s,a))
        td_target = r + self.gamma * best_next
        self.Q[s, a] += self.alpha * (td_target - self.Q[s, a])
    
    def train(self, episodes=300, steps_per_episode=500):
        all_rewards = []
        avg_skills = []
        min_skills = []
        std_skills = []
        
        for ep in range(episodes):
            self.env.reset()
            state_bin = self.discretize_state(self.env.k)
            
            episode_rewards = []
            
            for step in range(steps_per_episode):
                action = self.select_action(state_bin)
                # Ensure your env.step returns: next_state, reward, done, info
                _, reward, _, _ = self.env.step(action)
                
                next_state_bin = self.discretize_state(self.env.k)
                self.update(state_bin, action, reward, next_state_bin)
                
                state_bin = next_state_bin
                episode_rewards.append(reward)
                
            # decay epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            
            avg_ep_reward = np.mean(episode_rewards)
            final_skill = np.mean(self.env.k)
            
            all_rewards.append(avg_ep_reward)
            avg_skills.append(np.mean(self.env.k))
            min_skills.append(np.min(self.env.k))
            std_skills.append(np.std(self.env.k))
            
            if (ep + 1) % 10 == 0:
                print(f"Episode {ep+1} | Reward: {avg_ep_reward:.4f} | Skill: {final_skill:.4f} | Epsilon: {self.epsilon:.3f}")
                
        return all_rewards, avg_skills, min_skills, std_skills

def smooth(x, window=5):
    return np.convolve(x, np.ones(window)/window, mode='valid')

def plot_results(rewards, skills):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title("Avg Reward per Episode")
    plt.xlabel("Episode")
    
    plt.subplot(1, 2, 2)
    plt.plot(smooth(skills))
    plt.title("Final Avg Skill per Episode")
    plt.xlabel("Episode")
    
    plt.tight_layout()
    plt.savefig("figs/q_learning_agent.png")
    plt.show()

if __name__ == "__main__":
    # Initialize the agent
    agent = QLearningAgent(n_bins=10, alpha=0.1, gamma=0.95)
    
    # Run training
    rewards, skills, min_skills, std_skills = agent.train(episodes=300, steps_per_episode=500)
    
    # Plotting
    plot_results(rewards, skills)
