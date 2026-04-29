# representations/compare_visual.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from typing_env import TypingEnv
from visual_env_runner import step_with_logging
from agents_wrapper import rule_action, QAgentWrapper, DQNWrapper, ReinforceWrapper, ActorCriticWrapper


# def _render_frame(fig, axes, logs, avg_hist, min_hist, bigrams, titles, frame_idx):
#     for ax in axes.flat:
#         ax.clear()

#     # ===============================
#     # ROW 1 -> BAR PLOTS
#     # ===============================
#     for i, (log, title) in enumerate(zip(logs, titles)):
#         ax = axes[0, i]

#         k = log["new_k"]
#         delta = log["delta_k"]
#         target = log["bigram_id"]

#         x = np.arange(len(bigrams))

#         colors = []
#         for j in range(len(k)):
#             if j == target:
#                 colors.append("blue")
#             elif delta[j] > 0:
#                 colors.append("green")
#             else:
#                 colors.append("red")

#         ax.bar(x, k, color=colors)
#         ax.set_xticks(x)
#         ax.set_xticklabels(bigrams, rotation=90, fontsize=6)
#         ax.set_ylim(0, 1)
#         ax.set_title(
#             f"{title}\n"
#             f"Target: {log['bigram']} | "
#             f"Avg: {np.mean(k):.2f} | Min: {np.min(k):.2f}"
#         )

#     # ===============================
#     # ROW 2 -> AVG SKILL
#     # ===============================
#     for i in range(3):
#         ax = axes[1, i]
#         ax.plot(avg_hist[i][: frame_idx + 1], label=titles[i])
#         ax.set_ylim(0, 1)
#         ax.set_title(f"{titles[i]} - Avg Skill")
#         ax.grid(alpha=0.3)

#     # ===============================
#     # ROW 3 -> MIN SKILL
#     # ===============================
#     for i in range(3):
#         ax = axes[2, i]
#         ax.plot(min_hist[i][: frame_idx + 1], label=titles[i])
#         ax.set_ylim(0, 1)
#         ax.set_title(f"{titles[i]} - Min Skill")
#         ax.grid(alpha=0.3)

#     fig.suptitle(
#         f"Comparision of Agents | Step {frame_idx + 1}",
#         fontsize=16,
#     )
#     fig.tight_layout(rect=[0, 0, 1, 0.96])


def _render_frame(fig, axes, logs, bigrams, titles, frame_idx):
    for ax in axes:
        ax.clear()

    for i, (log, title) in enumerate(zip(logs, titles)):
        ax = axes[i]

        k = log["new_k"]
        delta = log["delta_k"]
        target = log["bigram_id"]

        x = np.arange(len(bigrams))

        colors = []
        for j in range(len(k)):
            if j == target:
                colors.append("blue")
            elif delta[j] > 0:
                colors.append("green")
            else:
                colors.append("red")

        ax.bar(x, k, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(bigrams, rotation=90, fontsize=6)
        ax.set_ylim(0, 1)

        ax.set_title(
            f"{title}\n"
            f"Target: {log['bigram']} | "
            f"Avg: {np.mean(k):.2f} | Min: {np.min(k):.2f}"
        )

    fig.suptitle(f"Agent Comparison | Step {frame_idx + 1}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

def run_multi_agent(steps=20, save_path=None, show=True, fps=4):
    env_rule = TypingEnv()
    env_q = TypingEnv()
    env_dqn = TypingEnv()
    env_reinforce = TypingEnv()
    env_actor_critic = TypingEnv()

    env_rule.reset()
    env_q.reset()
    env_dqn.reset()
    env_reinforce.reset()
    env_actor_critic.reset()

    q_agent = QAgentWrapper()
    dqn_agent = DQNWrapper()
    reinforce_agent = ReinforceWrapper()
    actor_critic_agent = ActorCriticWrapper()

    bigrams = env_rule.bigrams

    # ---- history tracking ----
    # avg_hist = [[], [], []]
    # min_hist = [[], [], []]
    step_logs = []

    titles = ["Rule-Based", "Q-Learning", "DQN", "REINFORCE", "Actor-Critic"]

    for step in range(steps):
        # -------- RUN STEP --------
        logs = []

        # Rule
        a_rule = rule_action(env_rule)
        logs.append(step_with_logging(env_rule, a_rule))

        # Q
        a_q = q_agent.get_action(env_q)
        logs.append(step_with_logging(env_q, a_q))

        # DQN
        a_dqn = dqn_agent.get_action(env_dqn)
        logs.append(step_with_logging(env_dqn, a_dqn))
        
        # REINFORCE
        a_rf = reinforce_agent.get_action(env_reinforce)
        logs.append(step_with_logging(env_reinforce, a_rf))

        # Actor-Critic
        a_ac = actor_critic_agent.get_action(env_actor_critic)
        logs.append(step_with_logging(env_actor_critic, a_ac))

        # -------- UPDATE METRICS --------
        # for i, log in enumerate(logs):
        #     k = log["new_k"]
        #     avg_hist[i].append(np.mean(k))
        #     min_hist[i].append(np.min(k))

        step_logs.append(logs)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))

    def update(frame_idx):
        # _render_frame(fig, axes, step_logs[frame_idx], avg_hist, min_hist, bigrams, titles, frame_idx)
        _render_frame(fig, axes, step_logs[frame_idx], bigrams, titles, frame_idx)
        return []


    animation = FuncAnimation(
        fig,
        update,
        frames=len(step_logs),
        interval=250,
        blit=False,
        repeat=False,
    )

    if save_path:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        animation.save(str(output_path), writer=PillowWriter(fps=fps))

    if show:
        plt.show()
    else:
        plt.close(fig)

    return animation


if __name__ == "__main__":
    run_multi_agent(steps=100, save_path="figs/compare_visuals_reward_tuning.gif")