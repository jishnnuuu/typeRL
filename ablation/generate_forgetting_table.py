import pickle
import pandas as pd
import numpy as np


NAME_MAP = {
    "linear": "Linear Forgetting",
    "log": "Logarithmic Forgetting",
}


def summarize(results):
    rows = []

    for agent, configs in results.items():
        for name, res in configs.items():

            avg_final = np.mean(res["avg"][-10:])
            min_final = np.mean(res["min"][-10:])
            std_final = np.mean(res["std"][-10:])

            rows.append({
                "Agent": agent,
                "Forgetting": NAME_MAP[name],
                "Avg Skill": avg_final,
                "Min Skill": min_final,
                "Std Dev": std_final,
            })

    return pd.DataFrame(rows)


def main():
    with open("ablation/results/forgetting_ablation.pkl", "rb") as f:
        results = pickle.load(f)

    df = summarize(results)

    print("\n===== FORGETTING ABLATION =====")
    print(df.round(4))

    print("\n===== LATEX =====")
    print(df.to_latex(index=False, float_format="%.4f"))

    df.to_csv("ablation/results/forgetting_ablation.csv", index=False)


if __name__ == "__main__":
    main()