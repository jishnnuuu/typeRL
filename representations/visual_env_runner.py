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
from text_processing import count_tracked_bigrams, counts_to_vector


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


# custom step for full visibility into the environment's internal workings
def step_with_logging(env, action):
    bigram_id, difficulty = env.decode_action(action)
    
    # ---- Sentence ----
    sentence = env.sample_sentence(bigram_id, difficulty)
    
    # ---- Counts ----
    counts_dict = count_tracked_bigrams(sentence)
    counts = counts_to_vector(counts_dict)
    
    # ---- Accuracy ----
    acc = env.simulate_accuracy(counts, difficulty)
    
    # ---- Before ----
    prev_k = env.k.copy()
    
    # ---- Apply updates ----
    env.update_skills(counts, acc)
    env.update_timers(counts)
    
    # ---- After ----
    new_k = env.k.copy()
    
    # ---- Changes ----
    delta_k = new_k - prev_k
    
    return {
        "bigram_id": bigram_id,
        "bigram": env.bigrams[bigram_id],
        "difficulty": difficulty,
        "sentence": sentence,
        "counts": counts,
        "accuracy": acc,
        "prev_k": prev_k,
        "new_k": new_k,
        "delta_k": delta_k,
    }


# rule based
def run_episode(steps=20):
    env = TypingEnv()
    
    print("\n--- Starting Episode ---\n")
    
    state = env.reset()
    
    logs = []
    
    for step in range(steps):
        action = select_rule_action(env)
        
        log = step_with_logging(env, action)
        
        log["step"] = step
        log["avg_skill"] = np.mean(log["new_k"])
        
        logs.append(log)
        
        # ---- Debug Print ----
        print(f"Step {step}")
        print(f"  Target Bigram : {log['bigram']}")
        print(f"  Difficulty    : {log['difficulty']}")
        print(f"  Sentence      : {log['sentence'][:60]}...")
        
        # Show only active bigrams
        active = np.where(log["counts"] > 0)[0]
        
        print("  Active Bigrams (count, acc, delta):")
        for b in active:
            print(
                f"    {env.bigrams[b]} | "
                f"count={int(log['counts'][b])} | "
                f"acc={log['accuracy'][b]:.2f} | "
                f"Δ={log['delta_k'][b]:.4f}"
            )
            
        print(f"  Avg Skill: {log['avg_skill']:.4f}")
        print("-" * 50)
        
    return logs


# run script
if __name__ == "__main__":
    logs = run_episode(steps=10)