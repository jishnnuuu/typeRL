# representations/visualizer.py

import matplotlib.pyplot as plt
import numpy as np


def plot_step(log, bigrams):
    k = log["new_k"]
    delta = log["delta_k"]
    target = log["bigram_id"]
    
    x = np.arange(len(bigrams))
    
    # ---- Colors ----
    colors = []
    for i in range(len(k)):
        if i == target:
            colors.append("blue")  # selected bigram
        elif delta[i] > 0:
            colors.append("green")  # learned
        else:
            colors.append("red")  # forgot
            
    # ---- Plot ----
    plt.figure(figsize=(12, 5))
    plt.bar(x, k, color=colors)
    
    plt.xticks(x, bigrams, rotation=90, fontsize=6)
    plt.xlabel("Bigrams")
    plt.ylabel("Skill (k)")
    plt.title(f"Step {log['step']} | Target: {log['bigram']}")
    
    plt.tight_layout()
    plt.show()