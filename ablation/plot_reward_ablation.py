import pickle
import numpy as np
import matplotlib.pyplot as plt


def smooth(x, window=10):
    return np.convolve(x, np.ones(window)/window, mode='same')


def plot_metric(results, agent, metric, title):
    plt.figure(figsize=(10, 5))

    for name, res in results[agent].items():
        plt.plot(smooth(res[metric]), label=name)

    plt.title(f"{agent} - {title}")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"ablation/results/{agent}_{metric}.png")
    plt.show()


def main():
    with open("ablation/results/reward_ablation.pkl", "rb") as f:
        results = pickle.load(f)

    for agent in ["DQN", "PPO"]:
        plot_metric(results, agent, "avg", "Average Skill")
        plot_metric(results, agent, "min", "Minimum Skill")
        plot_metric(results, agent, "std", "Skill Std")


if __name__ == "__main__":
    main()