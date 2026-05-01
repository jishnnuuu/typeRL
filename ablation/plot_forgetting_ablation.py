import pickle
import numpy as np
import matplotlib.pyplot as plt


def smooth(x, window=10):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode='same')


NAME_MAP = {
    "linear": "Linear Forgetting",
    "log": "Logarithmic Forgetting",
}


def plot_metric(results, agent, metric, title):
    plt.figure(figsize=(8, 5))

    for name, res in results[agent].items():
        label = NAME_MAP.get(name, name)
        plt.plot(smooth(res[metric]), label=label)

    plt.title(f"{agent} - {title}")
    plt.xlabel("Episodes")
    plt.ylabel(title)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"ablation/results/{agent}_{metric}_forgetting.png")
    plt.show()


def plot_all(results):
    for agent in ["DQN", "PPO"]:
        plot_metric(results, agent, "avg", "Average Skill")
        plot_metric(results, agent, "min", "Minimum Skill")
        plot_metric(results, agent, "std", "Skill Std Deviation")


def main():
    with open("ablation/results/forgetting_ablation.pkl", "rb") as f:
        results = pickle.load(f)

    plot_all(results)


if __name__ == "__main__":
    main()