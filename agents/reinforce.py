import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from typing_env import TypingEnv

import matplotlib.pyplot as plt


# -------------------------------
# Policy Network
# -------------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)


# -------------------------------
# REINFORCE Agent
# -------------------------------
class ReinforceAgent:
    def __init__(
        self,
        lr=1e-3,
        gamma=0.99
    ):
        self.env = TypingEnv()
        
        self.state_dim = len(self.env.get_state())
        self.action_dim = self.env.K * self.env.L
        
        self.policy = PolicyNetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma = gamma

    # sample action from policy
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state_tensor)
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action)

    # compute discounted returns
    def compute_returns(self, rewards):
        returns = []
        G = 0
        
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
            
        return returns

    # training loop
    def train(self, episodes=300):
        all_rewards = []
        all_skills = []
        
        for ep in range(episodes):
            state = self.env.reset()
            
            log_probs = []
            rewards = []
            
            done = False
            
            while not done:
                action, log_prob = self.select_action(state)
                
                next_state, reward, done, _ = self.env.step(action)
                
                log_probs.append(log_prob)
                rewards.append(reward)
                
                state = next_state

            # compute returns
            returns = self.compute_returns(rewards)
            returns = torch.FloatTensor(returns)
            
            # normalize returns (important for stability)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # policy loss
            loss = 0
            for log_prob, G in zip(log_probs, returns):
                loss += -log_prob * G
            
            # update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            avg_reward = np.mean(rewards)
            final_skill = np.mean(self.env.k)
            
            all_rewards.append(avg_reward)
            all_skills.append(final_skill)
            
            if (ep + 1) % 10 == 0:
                print(f"Episode {ep+1} | Reward: {avg_reward:.4f} | Skill: {final_skill:.4f}")
        
        return all_rewards, all_skills
    
    def save(self, path="models/reinforce.pth"):
        torch.save(self.policy.state_dict(), path)

    def load(self, path="models/reinforce.pth"):
        self.policy.load_state_dict(torch.load(path))
        self.policy.eval()

def plot_results(rewards, skills):
    plt.figure(figsize=(10,5))
    
    plt.subplot(1,2,1)
    plt.plot(rewards)
    plt.title("Reward")

    plt.subplot(1,2,2)
    plt.plot(skills)
    plt.title("Skill")

    plt.tight_layout()
    plt.savefig("figs/reinforce.png")
    plt.show()


if __name__ == "__main__":
    agent = ReinforceAgent()
    rewards, skills = agent.train(episodes=300)
    agent.save()
    plot_results(rewards, skills)