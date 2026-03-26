import time
import numpy as np

from typing_env import TypingEnv
from dqn_agent import DQNAgent
from text_processing import count_tracked_bigrams, counts_to_vector

from visualizer import LiveVisualizer


class TypingTutor:
    def __init__(self):
        self.env = TypingEnv()
        self.agent = DQNAgent()

        self.visualizer = LiveVisualizer(self.env.bigrams)

    def compute_accuracy(self, target, typed):
        correct = sum(1 for t, u in zip(target, typed) if t == u)
        return correct / max(len(target), 1)

    def compute_wpm(self, typed, time_taken):
        words = len(typed) / 5
        minutes = time_taken / 60
        return words / max(minutes, 1e-6)

    def run(self, steps=15):
        state = self.env.reset()

        print("\n🚀 RL Typing Tutor (LIVE DASHBOARD)\n")

        for step in range(steps):
            print("=" * 60)
            print(f"Step {step+1}")

            action = self.agent.select_action(state)
            bigram_id, difficulty = self.env.decode_action(action)

            target_bigram = self.env.bigrams[bigram_id]
            sentence = self.env.sample_sentence(bigram_id, difficulty)

            print(f"\n🎯 Target Bigram: {target_bigram}")
            print(f"⚙️ Difficulty: {difficulty}")
            print("\n📌 Sentence:")
            print(sentence)

            start = time.time()
            user_input = input("\n⌨️ Your Input: ")
            end = time.time()

            time_taken = end - start

            acc = self.compute_accuracy(sentence, user_input)
            wpm = self.compute_wpm(user_input, time_taken)

            print(f"\nAccuracy: {acc:.2f} | WPM: {wpm:.2f}")

            # -----------------------------
            # Bigram updates
            # -----------------------------
            counts_dict = count_tracked_bigrams(user_input)
            counts = counts_to_vector(counts_dict)

            acc_vec = np.zeros(self.env.K)
            acc_vec[counts > 0] = acc

            prev_skill = np.mean(self.env.k)

            self.env.update_skills(counts, acc_vec)
            self.env.update_timers(counts)

            new_skill = np.mean(self.env.k)

            print(f"Skill: {prev_skill:.3f} → {new_skill:.3f}")

            weakest = np.argmin(self.env.k)
            print(f"⚠️ Weakest: {self.env.bigrams[weakest]}")

            # -----------------------------
            # LIVE VISUAL UPDATE
            # -----------------------------
            self.visualizer.update(self.env.k, acc, wpm)

            state = self.env.get_state()

        print("\n🎉 Session Complete\n")

if __name__ == "__main__":
    tutor = TypingTutor()
    tutor.run()