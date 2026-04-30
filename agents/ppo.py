import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing_env import TypingEnv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ==========================================
# PPO NETWORK
# ==========================================
class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
        )

        self.policy = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.shared(x)
        logits = self.policy(x)
        value = self.value(x)

        probs = torch.softmax(logits/1.5, dim=-1)
        return probs, value


# ==========================================
# PPO AGENT
# ==========================================
class PPOAgent:
    def __init__(self, env, lr=1e-4, gamma=0.99, lam=0.95, clip_eps=0.15,
                epochs=4, batch_size=64, entropy_coef=0.1):

        self.env = env

        self.state_dim = len(env.get_state())
        self.action_dim = env.K * env.L

        self.model = PPONetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef

    # ==========================================
    # COLLECT TRAJECTORY
    # ==========================================
    def collect_trajectory(self):
        states, actions, log_probs = [], [], []
        rewards, values, dones = [], [], []

        state = self.env.reset()
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            probs, value = self.model(state_tensor)
            dist = torch.distributions.Categorical(probs)

            action = dist.sample()

            next_state, reward, done, _ = self.env.step(action.item())

            states.append(state)
            actions.append(action.item())
            log_probs.append(dist.log_prob(action).item())
            rewards.append(reward)
            values.append(value.item())
            dones.append(done)

            state = next_state
        # after loop
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        _, last_value = self.model(state_tensor)

        return states, actions, log_probs, rewards, values, dones, last_value.item()

    # ==========================================
    # GAE (ADVANTAGE ESTIMATION)
    # ==========================================
    def compute_gae(self, rewards, values, done, last_value):
        advantages = []
        gae = 0
        next_value = last_value

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - done[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - done[t]) * gae
            advantages.insert(0, gae)
            next_value = values[t]

        returns = np.array(advantages) + np.array(values)
        return returns, np.array(advantages)

    # ==========================================
    # PPO UPDATE
    # ==========================================
    def update(self, states, actions, old_log_probs, returns, advantages):
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        old_log_probs = torch.FloatTensor(np.array(old_log_probs))
        returns = torch.FloatTensor(np.array(returns))
        advantages = torch.FloatTensor(np.array(advantages))

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.clamp(advantages, -5, 5)

        dataset_size = len(states)

        for _ in range(self.epochs):
            indices = np.random.permutation(dataset_size)

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                s = states[batch_idx]
                a = actions[batch_idx]
                old_lp = old_log_probs[batch_idx]
                ret = returns[batch_idx]
                adv = advantages[batch_idx]

                probs, values = self.model(s)
                dist = torch.distributions.Categorical(probs)

                new_log_probs = dist.log_prob(a)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_lp)

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (ret - values.squeeze()).pow(2).mean()

                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

    # ==========================================
    # TRAIN LOOP
    # ==========================================
    def train(self, episodes=500):
        reward_history = []
        skill_history = []
        min_skill_history = []
        std_skill_history = []

        for ep in range(episodes):
            states, actions, log_probs, rewards, values, dones, last_value = self.collect_trajectory()
            
            rewards = np.array(rewards)

            returns, advantages = self.compute_gae(rewards, values, dones, last_value)

            self.update(states, actions, log_probs, returns, advantages)

            avg_rewards = np.mean(rewards)
            avg_skill = np.mean(self.env.k)  # <-- track skill  

            reward_history.append(avg_rewards)
            skill_history.append(avg_skill)
            min_skill_history.append(np.min(self.env.k))
            std_skill_history.append(np.std(self.env.k))

            if (ep + 1) % 10 == 0:
                print(f"Episode {ep+1} | Reward: {avg_rewards:.2f} | Skill: {avg_skill:.4f}")

        return reward_history, skill_history, min_skill_history, std_skill_history

    def save(self, path="models/ppo_model.pth"):
        torch.save(self.model.state_dict(), path)

def plot_results(rewards, skills):
    os.makedirs("figs", exist_ok=True)
    plt.figure(figsize=(12, 5))

    # -------- Reward plot --------
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3)

    if len(rewards) > 10:
        smooth = np.convolve(rewards, np.ones(10)/10, mode='valid')
        plt.plot(smooth)

    plt.title("Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    # -------- Skill plot --------
    plt.subplot(1, 2, 2)
    plt.plot(skills)

    if len(skills) > 10:
        smooth = np.convolve(skills, np.ones(10)/10, mode='valid')
        plt.plot(smooth)

    plt.title("Typing Skill Level")
    plt.xlabel("Episode")
    plt.ylabel("Skill")

    plt.tight_layout()
    plt.savefig("figs/ppo_results.png")
    plt.show()


# ==========================================
# USAGE
# ==========================================
if __name__ == "__main__":
    env = TypingEnv()
    agent = PPOAgent(env)

    rewards, skills, min_skills, std_skills = agent.train(episodes=500)
    plot_results(rewards, skills)  
    agent.save()
