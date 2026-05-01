import pickle
import pandas as pd
import numpy as np

NAME_MAP = {
    "delta_only": "Delta Only",
    "accuracy_only": "Accuracy Only",
    "weak_avg_only": "Weak Avg Only",
    "timer_only": "Timer Only",
    "std_only": "Std Dev Only",
    "full": "Combined Reward",
}

def summarize(results):
    rows = []

    for agent, configs in results.items():
        for name, res in configs.items():

            avg_final = np.mean(res["avg"][-10:])
            min_final = np.mean(res["min"][-10:])
            std_final = np.mean(res["std"][-10:])
            reward_final = res["reward"][-1]

            rows.append({
                "Agent": agent,
                "Reward": name,
                "Avg Skill": avg_final,
                "Min Skill": min_final,
                "Std Dev": std_final,
                "Reward": reward_final,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["Agent", "Experiment"])
    return df

def summarize(results):
    rows = []

    for agent, configs in results.items():
        for name, res in configs.items():

            avg_final = np.mean(res["avg"][-10:])
            min_final = np.mean(res["min"][-10:])
            std_final = np.mean(res["std"][-10:])

            rows.append({
                "Agent": agent,
                "Experiment": NAME_MAP.get(name, name),   # <-- KEY CHANGE
                "Avg Skill": avg_final,
                "Min Skill": min_final,
                "Std Dev": std_final,
            })

    return pd.DataFrame(rows)


def highlight_best(df):
    # For each agent, highlight best rows
    df_copy = df.copy()

    for agent in df["Agent"].unique():
        mask = df["Agent"] == agent

        # best avg, min, lowest std
        best_avg = df.loc[mask, "Avg Skill"].max()
        best_min = df.loc[mask, "Min Skill"].max()
        best_std = df.loc[mask, "Std Dev"].min()

        df_copy.loc[mask & (df["Avg Skill"] == best_avg), "Avg Skill"] = \
            df_copy["Avg Skill"].astype(str) + " *"

        df_copy.loc[mask & (df["Min Skill"] == best_min), "Min Skill"] = \
            df_copy["Min Skill"].astype(str) + " *"

        df_copy.loc[mask & (df["Std Dev"] == best_std), "Std Dev"] = \
            df_copy["Std Dev"].astype(str) + " *"

    return df_copy


def main():
    with open("ablation/results/reward_ablation.pkl", "rb") as f:
        results = pickle.load(f)

    df = summarize(results)
    df_highlight = highlight_best(df)

    print("\n===== RAW TABLE =====")
    print(df.round(4))

    print("\n===== HIGHLIGHTED TABLE =====")
    print(df_highlight)

    # save
    df.to_csv("ablation/results/reward_ablation.csv", index=False)


if __name__ == "__main__":
    main()