import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from typing_env import TypingEnv

# neural network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
    def forward(self, x):
        return self.net(x)


# replay buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, s, a, r, s_next):
        self.buffer.append((s, a, r, s_next))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next = zip(*batch)
        return (
            np.array(s),
            np.array(a),
            np.array(r),
            np.array(s_next),
        )
        
    def __len__(self):
        return len(self.buffer)

# DQN agent
class DQNAgent:
    def __init__(
        self,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        batch_size=64,
        target_update=10,
    ):
        self.env = TypingEnv()
        
        self.state_dim = len(self.env.get_state())
        self.action_dim = self.env.K * self.env.L
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # networks
        self.q_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # replay buffer
        self.buffer = ReplayBuffer()

        # hyperparams
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update

    # action selection
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return q_values.argmax().item()

    # training step
    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return
        s, a, r, s_next = self.buffer.sample(self.batch_size)
        
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        s_next = torch.FloatTensor(s_next).to(self.device)
        
        # current Q
        q_values = self.q_net(s)
        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        
        # target Q
        with torch.no_grad():
            next_q = self.target_net(s_next).max(1)[0]
            target = r + self.gamma * next_q
            
        # loss
        loss = nn.MSELoss()(q_value, target)
        
        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # training loop
    def train(self, episodes=100, steps_per_episode=200):
        all_rewards = []
        all_skills = []
        
        for ep in range(episodes):
            state = self.env.reset()
            episode_rewards = []
            
            for step in range(steps_per_episode):
                action = self.select_action(state)
                next_state, reward, _, _ = self.env.step(action)
                self.buffer.push(state, action, reward, next_state)
                self.train_step()
                
                state = next_state
                episode_rewards.append(reward)
                
            # epsilon decay
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            
            # update target network
            if ep % self.target_update == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
                
            avg_reward = np.mean(episode_rewards)
            final_skill = np.mean(self.env.k)
            
            all_rewards.append(avg_reward)
            all_skills.append(final_skill)
            print(f"Ep {ep+1} | Reward: {avg_reward:.4f} | Skill: {final_skill:.4f} | Eps: {self.epsilon:.3f}")
        return all_rewards, all_skills

import matplotlib.pyplot as plt

def plot_dqn_results(rewards, skills):
    """
    Visualizes the training progress of the DQN agent.
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Average Reward per Episode
    ax1.plot(rewards, color='#1f77b4', label='Episode Reward')
    # Add a moving average to see the trend through the noise
    if len(rewards) > 10:
        moving_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
        ax1.plot(range(9, len(rewards)), moving_avg, color='orange', linewidth=2, label='10-Ep Moving Avg')
    
    ax1.set_title('Agent Reward Trend', fontsize=14)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Avg Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Skill Progression
    ax2.plot(skills, color='#2ca02c', linewidth=2)
    ax2.set_title('Typing Skill Progression', fontsize=14)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Mean Skill (k-vector)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figs/dqn_agent.png")
    plt.show()

# --- Execution Example ---
if __name__ == "__main__":
    # Initialize the agent
    agent = DQNAgent()
    
    # Run the training (this uses your existing train method)
    rewards, skills = agent.train(episodes=100, steps_per_episode=300)
    
    # Generate the plots
    plot_dqn_results(rewards, skills)