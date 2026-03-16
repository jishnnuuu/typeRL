import numpy as np
import matplotlib.pyplot as plt

from typing_env import TypingEnv


env = TypingEnv()

state = env.reset()

target_bigram = 0
difficulty = 1
action = target_bigram * env.L + difficulty

steps = 200
skill_history = []

for step in range(steps):
    state, reward, done, _ = env.step(action)
    skill_history.append(env.k[target_bigram])

plt.figure(figsize=(8,5))
plt.plot(skill_history)
plt.xlabel("Training Step")
plt.ylabel("Skill Level")
plt.title(f"Learning Curve for Bigram {env.bigrams[target_bigram]}")
plt.grid(True)
plt.show()