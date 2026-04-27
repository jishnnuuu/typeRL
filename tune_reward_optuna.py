import optuna
import numpy as np

from typing_env import TypingEnv


# -------------------------------
# Simple policy (use rule-based)
# -------------------------------
def select_action(env):
    scores = env.k - 0.1 * env.t
    b = np.argmin(scores)
    skill = env.k[b]
    
    if skill < 0.4:
        d = 0
    elif skill < 0.6:
        d = 1
    elif skill < 0.75:
        d = 2
    elif skill < 0.9:
        d = 3
    else:
        d = 4
    return b * env.L + d


# -------------------------------
# Evaluation function
# -------------------------------
def evaluate(weights, episodes=5):
    scores = []

    for _ in range(episodes):
        env = TypingEnv(reward_weights=weights)
        state = env.reset()

        done = False
        while not done:
            action = select_action(env)
            state, _, done, _ = env.step(action)

        avg_k = np.mean(env.k)
        min_k = np.min(env.k)
        std_k = np.std(env.k)

        final_score = (
            0.6 * avg_k +
            0.3 * min_k -
            0.1 * std_k
        )

        scores.append(final_score)

    return np.mean(scores)


# -------------------------------
# Optuna objective
# -------------------------------
def objective(trial):
    weights = {
        "delta_skill": trial.suggest_float("delta_skill", 1.0, 5.0),
        "accuracy": trial.suggest_float("accuracy", 0.0, 1.0),
        "weak_avg": trial.suggest_float("weak_avg", 1.0, 5.0),
        "timer_penalty": trial.suggest_float("timer_penalty", 0.0, 2.0),
        "std_penalty": trial.suggest_float("std_penalty", 0.0, 1.5),
    }

    score = evaluate(weights)

    return score


# -------------------------------
# Run optimization
# -------------------------------
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    
    study.optimize(objective, n_trials=100)

    print("\nBest weights:")
    print(study.best_params)

    print("\nBest score:")
    print(study.best_value)