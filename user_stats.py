import numpy as np

class UserStats:
    def __init__(self):
        self.skill_history = []
        self.accuracy_history = []
        self.wpm_history = []

    def update(self, skill, acc, wpm):
        self.skill_history.append(skill)
        self.accuracy_history.append(acc)
        self.wpm_history.append(wpm)

    def summary(self):
        print("\n📊 Session Summary")
        print(f"Final Skill : {self.skill_history[-1]:.3f}")
        print(f"Avg Accuracy: {np.mean(self.accuracy_history):.2f}")
        print(f"Avg WPM     : {np.mean(self.wpm_history):.2f}")