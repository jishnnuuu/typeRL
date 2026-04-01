# TypeRL

TypeRL is an adaptive typing tutor powered by reinforcement learning.
Instead of selecting random sentences, it learns which character patterns a user struggles with and schedules targeted exercises to maximize long-term typing improvement.

---

## Overview

Most typing tutors rely on static or random sentence selection and do not adapt to individual weaknesses. TypeRL formulates exercise selection as a sequential decision-making problem and solves it using reinforcement learning.

At each step, the system decides:

* which bigram to practice,
* what difficulty level to use,
* when to revisit previously learned patterns.

Typing skill is represented as a vector of bigram mastery levels, and the agent learns a curriculum policy that improves performance across all patterns.

---

## Core Idea

Typing is decomposed into **bigrams** (character transitions), such as:

```
queen → qu, ue, ee, en
```

Instead of a single metric like WPM, the system models:

```
k = [k_th, k_er, k_qu, k_st, ...]
```

This enables:

* targeted practice,
* identification of weak patterns,
* personalized learning progression.

---

## Skill Model

* 40 high-frequency English bigrams are tracked
* Each bigram has a mastery score:
  $k_b \in [0, 1]$
* Skills improve with practice and decay over time

### Skill Update

$$
k_b \leftarrow \text{clip}\left(
k_b +
\alpha \cdot acc_b \cdot \log(1+c_b) \cdot (1-k_b)
--------------------------------------------------

\lambda \cdot (1-k_b) \cdot \log(1+t_b),
\ 0,\ 1
\right)
$$

| Symbol            | Meaning                      |
| ----------------- | ---------------------------- |
| $\alpha = 0.08$   | Learning rate                |
| $\lambda = 0.002$ | Forgetting rate              |
| $acc_b$           | Typing accuracy              |
| $c_b$             | Bigram frequency in sentence |
| $t_b$             | Time since last practice     |

---

## Reinforcement Learning Formulation

| Component    | Description                                            |
| ------------ | ------------------------------------------------------ |
| State        | $s = [\mathbf{k} | \mathbf{t}] \in \mathbb{R}^{80}$    |
| Action       | $(b, \ell)$ — bigram × difficulty (0–4)                |
| Action Space | 200 actions                                            |
| Reward       | $2\Delta\bar{k} + 0.3,acc + 0.3,\min(k) - 0.1,\bar{t}$ |

The reward encourages:

* overall improvement,
* focus on weakest skills,
* balanced coverage across all bigrams.

---

## Agents

### Rule-Based (Baseline)

A heuristic policy that selects the weakest bigram and assigns difficulty based on skill.

* No learning
* Interpretable baseline
* Ensures coverage

---

### Q-Learning

Tabular reinforcement learning agent.

* State: discretized mean skill
* Q-table: $(20, 200)$

**Limitation:** loses per-bigram information, reducing precision.

---

### DQN (Deep Q-Network)

Neural reinforcement learning agent using the full state.

```
Input (80)
 → Linear → ReLU
 → Linear(128) → ReLU
 → Linear(128)
 → Output (200)
```

| Parameter         | Value            |
| ----------------- | ---------------- |
| Optimizer         | Adam (1e-3)      |
| Replay buffer     | 10,000           |
| Batch size        | 64               |
| Target update     | every 5 episodes |
| Discount $\gamma$ | 0.99             |
| ε decay           | 0.995 → 0.05     |

---

## Environment Model

Typing performance is simulated using a logistic model:

$$
p_b = \sigma(k_b - d_\ell), \quad d_\ell = 0.2 \cdot \ell^{1.5}
$$

Each bigram occurrence is modeled as:

$$
X_i \sim \text{Bernoulli}(p_b), \quad acc_b = \frac{1}{c_b}\sum X_i
$$

This introduces realistic variability in typing behavior.

---

## Dataset

Sentences are generated using Llama 3.1 (via Groq API).

* 40 bigrams
* 5 difficulty levels
* 20 sentences per pair

Total: **4,000 sentences**

Each sentence:

* contains the target bigram at least 5 times
* is 8–12 words long
* maintains natural language structure

---

## Results

Each agent is trained for:

* 100 episodes
* 200 steps per episode

Metrics:

* average reward
* mean skill
* weakest skill

| Agent      | Avg Reward | Final Mean Skill |
| ---------- | ---------- | ---------------- |
| Rule-Based | —          | —                |
| Q-Learning | —          | —                |
| DQN        | —          | —                |

Training curves:

![Agent Comparison](figs/compare_agents.png)

---

## Project Structure

```
TypeRL/
├── bigrams.py
├── text_processing.py
├── generate_dataset.py
├── dataset_loader.py
├── typing_env.py
├── rule_based_agent.py
├── q_learning.py
├── dqn_agent.py
├── compare_agents.py
├── typing_component.py
├── typing_dataset_cleaned.csv
└── figs/
```

---

## Getting Started

### Install dependencies

```bash
pip install numpy torch matplotlib streamlit groq python-dotenv tenacity
```

### Generate dataset (optional)

```bash
echo "GROQ_API_KEY=your_key_here" > .env
python generate_dataset.py
```

### Train and compare agents

```bash
python compare_agents.py
```

### Run individual agents

```bash
python rule_based_agent.py
python q_learning.py
python dqn_agent.py
```

---

## Key Design Decisions

* Logarithmic forgetting prevents rapid skill collapse
* Nonlinear difficulty better models real typing complexity
* Minimum skill term ensures weak patterns are addressed
* LLM-generated dataset enables controlled training data
* Full-state DQN enables fine-grained curriculum learning

---

## Summary

TypeRL reframes typing practice as a learning problem.
Instead of fixed exercises, it learns how to teach by adapting to the user’s evolving skill profile.
