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
    
    

def animate_with_metrics(logs, bigrams, delay=0.3):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(14, 8))

    avg_history = []
    min_history = []

    for step, log in enumerate(logs):
        plt.clf()

        k = log["new_k"]
        delta = log["delta_k"]
        target = log["bigram_id"]

        avg_k = np.mean(k)
        min_k = np.min(k)

        avg_history.append(avg_k)
        min_history.append(min_k)

        x = np.arange(len(bigrams))

        # ---- Colors ----
        colors = []
        for i in range(len(k)):
            if i == target:
                colors.append("blue")
            elif delta[i] > 0:
                colors.append("green")
            else:
                colors.append("red")

        # =======================
        # Plot 1: Bar plot
        # =======================
        plt.subplot(2, 1, 1)
        plt.bar(x, k, color=colors)

        plt.xticks(x, bigrams, rotation=90, fontsize=6)
        plt.ylim(0, 1)

        plt.title(
            f"Step {log['step']} | Target: {log['bigram']} | "
            f"Avg: {avg_k:.3f} | Min: {min_k:.3f}"
        )

        plt.ylabel("Skill")

        # =======================
        # Plot 2: Metrics over time
        # =======================
        plt.subplot(2, 1, 2)
        plt.plot(avg_history, label="Average Skill")
        plt.plot(min_history, label="Minimum Skill")

        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.pause(delay)

    plt.show()