import numpy as np
from typing_env import TypingEnv


def diagnose_learning():
    env = TypingEnv()
    env.reset()
    
    bigram_id = 33
    print("\n--- Learning Dynamics ---\n")
    for step in range(20):
        action = bigram_id * env.L + 0  # easiest difficulty
        prev_k = env.k[bigram_id]
        state, reward, _, _ = env.step(action)
        new_k = env.k[bigram_id]
        delta = new_k - prev_k

        print(f"Step {step+1}")
        print(f"Skill before: {prev_k:.4f}")
        print(f"Skill after : {new_k:.4f}")
        print(f"Delta       : {delta:.6f}")
        print("-" * 30)


if __name__ == "__main__":
    diagnose_learning()