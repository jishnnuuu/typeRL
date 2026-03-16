# typing_env.py

import numpy as np
import random

from bigrams import BIGRAM_BAG
from text_processing import count_tracked_bigrams, counts_to_vector


class TypingEnv:
    def __init__(self):
        # bigram settings
        self.bigrams = BIGRAM_BAG
        self.K = len(self.bigrams)
        
        # difficulty levels
        self.L = 5
        
        # learning parameters
        self.alpha = 0.15      # learning rate
        self.lmbda = 0.004     # forgetting rate
        self.eta = 0.1
        
        # skill vector
        self.k = np.zeros(self.K)
        
        # timers (time since last practice)
        self.t = np.zeros(self.K)

    # reset environment to initial state
    def reset(self):
        # initialize skill levels randomly
        self.k = np.random.uniform(0.2, 0.3, size=self.K)
        
        # reset timers
        self.t = np.zeros(self.K)
        
        return self.get_state()

    # state representation: concatenate skill levels and timers
    def get_state(self):
        return np.concatenate([self.k, self.t])
    
    # decode action into bi-gram index and difficulty level
    def decode_action(self, action):
        bigram_id = action // self.L
        difficulty = action % self.L
        
        return bigram_id, difficulty
    
    # generate a random sentence containing the target bi-gram
    def generate_sentence(self, target_bigram):
        words = [
            "the","quick","brown","fox","jumps","over",
            "quiet","queen","quickly","quit","quiz",
            "river","stone","strong","story","storm"
        ]
        sentence = " ".join(random.choices(words, k=10))
        return sentence
    
    # sigmoid function for accuracy simulation
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # simulate typing accuracy based on skill levels and difficulty
    def simulate_accuracy(self, counts, difficulty):
        acc = np.zeros(self.K)
        difficulty_strength = difficulty * 0.15
        
        for b in range(self.K):
            c = int(counts[b])
            if c == 0:
                continue
            p = self.sigmoid(self.k[b] - difficulty_strength)
            trials = np.random.binomial(1, p, c)
            acc[b] = np.mean(trials)
        return acc
    
    #skill update rule based on learning and forgetting dynamics
    def update_skills(self, counts, acc):
        for b in range(self.K):
            c = counts[b]
            forget = self.lmbda * (1 - self.k[b]) * self.t[b]
            if c > 0:
                learn = self.alpha * acc[b] * np.log(1 + c) * (1 - self.k[b])
                self.k[b] += learn - forget
            else:
                self.k[b] -= forget
            # keep skills in valid range
            self.k[b] = np.clip(self.k[b], 0, 1)
            
    # timer update: reset to 0 if practiced, otherwise increment by 1
    def update_timers(self, counts):
        for b in range(self.K):
            if counts[b] > 0:
                self.t[b] = 0
            else:
                self.t[b] += 1
                
    # environment step: process action, update state, and return reward
    def step(self, action):
        bigram_id, difficulty = self.decode_action(action)
        sentence = self.generate_sentence(bigram_id)
        counts_dict = count_tracked_bigrams(sentence)
        counts = counts_to_vector(counts_dict)
        acc = self.simulate_accuracy(counts, difficulty)
        prev_avg_skill = np.mean(self.k)
        
        self.update_skills(counts, acc)
        self.update_timers(counts)
        
        new_avg_skill = np.mean(self.k)
        delta_skill = new_avg_skill - prev_avg_skill
        reward = delta_skill + self.eta * acc[bigram_id]
        
        next_state = self.get_state()
        done = False
        info = {}
        
        return next_state, reward, done, info