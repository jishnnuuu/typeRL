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


def animate_episode(logs, bigrams, delay=1):
    plt.figure(figsize=(12, 5))
    
    for log in logs:
        plt.clf()  # clear previous frame
        
        k = log["new_k"]
        delta = log["delta_k"]
        target = log["bigram_id"]
        
        x = np.arange(len(bigrams))
        
        # ---- Colors ----
        colors = []
        for i in range(len(k)):
            if i == target:
                colors.append("blue")  # selected
            elif delta[i] > 0:
                colors.append("green")  # learned
            else:
                colors.append("red")  # forgot
                
        # ---- Plot ----
        plt.bar(x, k, color=colors)
        
        plt.xticks(x, bigrams, rotation=90, fontsize=6)
        plt.ylim(0, 1)
        
        plt.title(
            f"Step {log['step']} | Target: {log['bigram']} | "
            f"Avg Skill: {log['avg_skill']:.3f}"
        )
        
        plt.text(
            0.01, 0.95,
            f"Difficulty: {log['difficulty']}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top'
        )
        
        plt.xlabel("Bigrams")
        plt.ylabel("Skill (k)")
        
        plt.tight_layout()
        
        plt.pause(delay)
        
    plt.show()