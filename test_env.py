import numpy as np
from typing_env import TypingEnv

env = TypingEnv()

state = env.reset()

print("State size:", len(state))

for step in range(10):
    action = np.random.randint(0, env.K * env.L)
    state, reward, done, _ = env.step(action)
    print("Step:", step, "Reward:", reward)