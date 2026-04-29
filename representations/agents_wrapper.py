# representations/agents_wrapper.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing_env import TypingEnv

import numpy as np
import torch

from agents.q_learning import QLearningAgent
from agents.dqn_agent import QNetwork

# rule based agent
def rule_action(env):
    scores = env.k - 0.1 * env.t
    weakest = np.argmin(scores)
    
    skill = env.k[weakest]
    
    if skill < 0.3:
        diff = 0
    elif skill < 0.5:
        diff = 1
    elif skill < 0.7:
        diff = 2
    elif skill < 0.85:
        diff = 3
    else:
        diff = 4
    return weakest * env.L + diff


# Q-learning agent wrapper
class QAgentWrapper:
    def __init__(self):
        self.agent = QLearningAgent()
        
    def get_action(self, env):
        state_bin = self.agent.discretize_state(env.k)
        return np.argmax(self.agent.Q[state_bin])


# DQN agent wrapper
class DQNWrapper:
    def __init__(self, model_path="models/dqn_model_best.pth"):
        self.env = None
        self.device = torch.device("cpu")
        
        # lazy init later
        self.model = None
        self.model_path = model_path
        
    def init_model(self, env):
        state_dim = len(env.get_state())
        action_dim = env.K * env.L
        
        self.model = QNetwork(state_dim, action_dim)
        self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        self.model.eval()
        
    def get_action(self, env):
        if self.model is None:
            self.init_model(env)
            
        state = env.get_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.model(state_tensor)
        
        return q_values.argmax().item()


from agents.reinforce import PolicyNetwork

class ReinforceWrapper:
    def __init__(self):
        self.env = TypingEnv()
        
        self.state_dim = len(self.env.get_state())
        self.action_dim = self.env.K * self.env.L
        
        self.policy = PolicyNetwork(self.state_dim, self.action_dim)
        self.policy.load_state_dict(torch.load("models/reinforce.pth"))
        self.policy.eval()

    def get_action(self, env):
        state = env.get_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            probs = self.policy(state_tensor)
        
        action = torch.argmax(probs).item()   # deterministic for visualization
        
        return action
    

import torch
from agents.actor_critic import ActorCriticNetwork


class ActorCriticWrapper:
    def __init__(self):
        self.env = TypingEnv()

        self.state_dim = len(self.env.get_state())
        self.action_dim = self.env.K * self.env.L

        self.model = ActorCriticNetwork(self.state_dim, self.action_dim)
        self.model.load_state_dict(torch.load("models/actor_critic.pth"))
        self.model.eval()

    def get_action(self, env):
        state = env.get_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            probs, _ = self.model(state_tensor)

        # deterministic for visualization
        return torch.argmax(probs).item()