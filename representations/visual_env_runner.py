""" 
visual_env_runner.py

- Get state
- Select action
- Run env.step()
- Log:
    - bigram
    - difficulty
    - accuracy
    - skill-vector
    - delta skill
"""

import os
import sys
# this makes sure we can import from the parent directory where typing_env.py is located
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing_env import TypingEnv


# rule-based agent action selection
def select_rule_action(env):
    scores = env.k - 0.1 * env.t
    weakest_bigram = np.argmin(scores)
    
    skill = env.k[weakest_bigram]
    
    if skill < 0.3:
        difficulty = 0
    elif skill < 0.5:
        difficulty = 1
    elif skill < 0.7:
        difficulty = 2
    elif skill < 0.85:
        difficulty = 3
    else:
        difficulty = 4
    
    action = weakest_bigram * env.L + difficulty
    return action


# rule based
def run_episode(steps=20):
    env = TypingEnv()
    
    print("\n--- Starting Episode ---\n")
    
    state = env.reset()
    
    logs = []
    
    for step in range(steps):
        # select action based on current state
        action = select_rule_action(env)
        bigram_id, difficulty = env.decode_action(action)
        
        # ---- Before Step ----
        prev_k = env.k.copy()
        prev_avg = np.mean(prev_k)
        
        # ---- Step ----
        next_state, reward, done, _ = env.step(action)
        
        # ---- After Step ----
        new_k = env.k.copy()
        new_avg = np.mean(new_k)
        
        delta = new_avg - prev_avg
        
        # ---- Log Info ----
        log = {
            "step": step,
            "bigram_id": bigram_id,
            "bigram": env.bigrams[bigram_id],
            "difficulty": difficulty,
            "reward": reward,
            "avg_skill": new_avg,
            "delta_skill": delta,
            "k_vector": new_k.copy(),
        }
        
        logs.append(log)
        
        # ---- Print (for debugging now) ----
        print(f"Step {step}")
        print(f"  Bigram     : {log['bigram']}")
        print(f"  Difficulty : {difficulty}")
        print(f"  Reward     : {reward:.4f}")
        print(f"  Avg Skill  : {new_avg:.4f}")
        print(f"  Delta Skill: {delta:.4f}")
        print("-" * 40)
    return logs


# run script
if __name__ == "__main__":
    logs = run_episode(steps=20)