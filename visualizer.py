import matplotlib.pyplot as plt
import numpy as np


class LiveVisualizer:
    def __init__(self, bigrams, top_n=10):
        self.bigrams = bigrams
        self.top_n = top_n

        plt.ion()  # interactive mode

        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))

        # Subplots
        self.ax_bar = self.axs[0, 0]
        self.ax_skill = self.axs[0, 1]
        self.ax_acc = self.axs[1, 0]
        self.ax_wpm = self.axs[1, 1]

        # histories
        self.skill_hist = []
        self.acc_hist = []
        self.wpm_hist = []

    def update(self, k, acc, wpm):
        avg_skill = np.mean(k)

        self.skill_hist.append(avg_skill)
        self.acc_hist.append(acc)
        self.wpm_hist.append(wpm)

        self.ax_bar.clear()
        self.ax_skill.clear()
        self.ax_acc.clear()
        self.ax_wpm.clear()

        # -----------------------------
        # 1️⃣ Weakest bigrams bar chart
        # -----------------------------
        idx = np.argsort(k)[:self.top_n]
        labels = [self.bigrams[i] for i in idx]
        values = k[idx]

        self.ax_bar.bar(labels, values)
        self.ax_bar.set_title("Weakest Bigrams")
        self.ax_bar.set_ylim(0, 1)

        # -----------------------------
        # 2️⃣ Skill progression
        # -----------------------------
        self.ax_skill.plot(self.skill_hist, marker='o')
        self.ax_skill.set_title("Avg Skill")
        self.ax_skill.set_ylim(0, 1)

        # -----------------------------
        # 3️⃣ Accuracy progression
        # -----------------------------
        self.ax_acc.plot(self.acc_hist, color='green')
        self.ax_acc.set_title("Accuracy")

        # -----------------------------
        # 4️⃣ WPM progression
        # -----------------------------
        self.ax_wpm.plot(self.wpm_hist, color='orange')
        self.ax_wpm.set_title("WPM")

        plt.tight_layout()
        plt.pause(0.1)