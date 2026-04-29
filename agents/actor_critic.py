import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Ensure environment is discoverable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing_env import TypingEnv

# -------------------------------
# Actor-Critic Network
# -------------------------------
class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
        )

        # Actor head (Policy)
        self.policy_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        # Critic head (Value)
        self.value_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        shared_features = self.shared(x)
        logits = self.policy_head(shared_features)
        value = self.value_head(shared_features)
        
        # We return logits for better numerical stability in distributions
        probs = torch.softmax(logits, dim=-1)
        return probs, value


# -------------------------------
# Actor-Critic Agent
# -------------------------------
class ActorCriticAgent:
    def __init__(self, lr=1e-4, gamma=0.99, entropy_coef=0.01):
        self.env = TypingEnv()

        self.state_dim = len(self.env.get_state())
        self.action_dim = self.env.K * self.env.L

        self.model = ActorCriticNetwork(self.state_dim, self.action_dim)
        # Using a slightly lower learning rate for Actor-Critic stability
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.entropy_coef = entropy_coef

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs, value = self.model(state_tensor)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action), value, dist.entropy()

    def train(self, episodes=300):
        all_rewards = []
        all_skills = []

        for ep in range(episodes):
            state = self.env.reset()
            episode_rewards = []
            done = False

            while not done:
                # 1. Select Action
                action, log_prob, value, entropy = self.select_action(state)

                # 2. Step Environment
                next_state, reward, done, _ = self.env.step(action)

                # 3. Estimate Next Value (V_t+1)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                _, next_value = self.model(next_state_tensor)

                # 4. TD Error (Advantage) Calculation
                # We detach next_value because we don't want to backprop through the target
                td_target = reward + self.gamma * next_value.detach() * (1 - int(done))
                advantage = td_target - value 

                # 5. Losses
                # Actor Loss: Negative log-prob * detached advantage
                actor_loss = -log_prob * advantage.detach()
                
                # Critic Loss: MSE between prediction and TD Target
                critic_loss = 0.5 * advantage.pow(2)

                # Total Loss with Entropy Bonus (encourages exploration)
                loss = actor_loss + critic_loss - (self.entropy_coef * entropy)

                # 6. Optimization Step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                state = next_state
                episode_rewards.append(reward)

            # Logging
            avg_reward = np.mean(episode_rewards)
            final_skill = np.mean(self.env.k)
            all_rewards.append(avg_reward)
            all_skills.append(final_skill)

            if (ep + 1) % 10 == 0:
                print(f"Ep {ep+1:3d} | Avg Reward: {avg_reward:7.4f} | Avg Skill: {final_skill:.4f}")

        return all_rewards, all_skills

    def save(self, path="models/actor_critic.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

# -------------------------------
# Visualization
# -------------------------------
def plot_results(rewards, skills):
    os.makedirs("figs", exist_ok=True)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards, color='blue', alpha=0.3)
    # Simple moving average for clarity
    if len(rewards) > 10:
        plt.plot(np.convolve(rewards, np.ones(10)/10, mode='valid'), color='blue')
    plt.title("Reward per Episode")
    plt.xlabel("Episode")

    plt.subplot(1, 2, 2)
    plt.plot(skills, color='green')
    plt.title("Typing Skill Level")
    plt.xlabel("Episode")

    plt.tight_layout()
    plt.savefig("figs/actor_critic_results.png")
    plt.show()

if __name__ == "__main__":
    agent = ActorCriticAgent(lr=5e-4) # Balanced LR
    rewards, skills = agent.train(episodes=300)
    plot_results(rewards, skills)
    agent.save()