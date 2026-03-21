import numpy as np
import matplotlib.pyplot as plt
from typing_env import TypingEnv


env = TypingEnv()
state = env.reset()

target_bigram = 0
difficulty = 1
action = target_bigram * env.L + difficulty


# learning curve experiment
steps = 200
learning_curve = []

env.reset()

for step in range(steps):
    state, reward, done, _ = env.step(action)
    learning_curve.append(env.k[target_bigram])


# forgetting curve experiment
env.reset()

learning_steps = 100
forget_steps = 100

forget_curve = []

# learn first
for step in range(learning_steps):
    state, reward, done, _ = env.step(action)

# stop practicing target bigram
for step in range(forget_steps):
    random_action = np.random.randint(0, env.K * env.L)
    state, reward, done, _ = env.step(random_action)
    forget_curve.append(env.k[target_bigram])

# difficulty comparison experiment
env.reset()

difficulty_results = []

for difficulty in range(env.L):
    env.reset()
    action = target_bigram * env.L + difficulty

    skill_progress = []

    for step in range(100):
        state, reward, done, _ = env.step(action)
        skill_progress.append(env.k[target_bigram])

    difficulty_results.append(skill_progress)



# plotting results
plt.figure(figsize=(15,4))

# Learning curve
plt.subplot(1,3,1)
plt.plot(learning_curve)
plt.title("Learning Curve")
plt.xlabel("Steps")
plt.ylabel("Skill")

# Forgetting curve
plt.subplot(1,3,2)
plt.plot(forget_curve)
plt.title("Forgetting Curve")
plt.xlabel("Steps")
plt.ylabel("Skill")

# Difficulty comparison
plt.subplot(1,3,3)

for i in range(env.L):
    plt.plot(difficulty_results[i], label=f"Difficulty {i}")

plt.title("Difficulty Effect")
plt.xlabel("Steps")
plt.ylabel("Skill")
plt.legend()

plt.tight_layout()
plt.show()