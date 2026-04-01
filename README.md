# TypeRL 🎯

An adaptive typing tutor powered by Reinforcement Learning. Instead of random sentence selection, TypeRL learns which bigrams you struggle with and schedules targeted exercises to maximize your long-term typing improvement.

---

## 📌 Overview

Most typing tutors (MonkeyType, Keybr, TypeRacer) use static or random sentence selection. TypeRL treats exercise selection as a **sequential decision-making problem** and solves it with RL — choosing *which* character pattern to practice, *how hard* to make it, and *when to revisit* previously learned patterns.

Typing skill is modeled as a vector of **bigram mastery levels** (e.g., `th`, `er`, `qu`, `st`). The RL agent learns a curriculum policy that maximizes cumulative skill improvement across all bigrams.

---

## 🧠 How It Works

### Bigram Skill Model
- **40 tracked bigrams** cover the most frequent English character transitions
- Each bigram has a mastery score $k_b \in [0, 1]$
- Skills **increase with practice** and **decay with neglect**

### Skill Update Rule
$$k_b \leftarrow \text{clip}\left(k_b + \underbrace{\alpha \cdot acc_b \cdot \log(1+c_b) \cdot (1-k_b)}_{\text{learning}} - \underbrace{\lambda \cdot (1-k_b) \cdot \log(1+t_b)}_{\text{forgetting}},\ 0,\ 1\right)$$

| Symbol | Meaning |
|--------|---------|
| $\alpha = 0.08$ | Learning rate |
| $\lambda = 0.002$ | Forgetting rate |
| $acc_b$ | Simulated typing accuracy for bigram $b$ |
| $c_b$ | Number of times bigram appeared in the sentence |
| $t_b$ | Steps since bigram was last practiced |

### MDP Formulation
| Component | Description |
|-----------|-------------|
| **State** | $s = [\mathbf{k} \| \mathbf{t}] \in \mathbb{R}^{80}$ — skill levels + practice timers |
| **Action** | $(b, \ell)$ — target bigram × difficulty level (0–4), 200 total actions |
| **Reward** | $2.0\,\Delta\bar{k} + 0.3\,acc_{b_t} + 0.3\,\min_b k_b - 0.1\,\bar{t}$ |

The reward encourages overall skill growth, penalizes neglecting the weakest bigram, and rewards consistent coverage across all patterns.

---

## 🤖 Agents

### Rule-Based (Baseline)
Greedy heuristic — always targets the bigram with the lowest score $k_b - 0.1 \cdot t_b$, and sets difficulty based on current skill. No training required.

### Q-Learning
Tabular RL agent. State is compressed to mean skill $\bar{k}$ and discretized into 20 bins. Q-table shape: $(20, 200)$.
- **Limitation:** Losing per-bigram information makes it hard to target specific weaknesses.

### DQN (Deep Q-Network)
Neural RL agent operating on the **full state** $\mathbb{R}^{80}$, using experience replay and a target network.

**Architecture:**
```
Input (80) → Linear → ReLU → Linear(128) → ReLU → Linear(128) → Output (200)
```

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam, lr = 1e-3 |
| Replay buffer | 10,000 |
| Batch size | 64 |
| Target update | Every 5 episodes |
| γ (discount) | 0.99 |
| ε decay | 0.995 → 0.05 |

---

## 📊 Results

> 100 episodes × 200 steps/episode on the same environment seed.

| Agent | Avg Reward | Final Mean Skill $\bar{k}$ |
|-------|-----------|--------------------------|
| Rule-Based | — | — |
| Q-Learning | — | — |
| DQN | — | — |

*Replace `—` with your actual numbers after running `compare_agents.py`.*

Training curves (reward and skill progression per episode):

![Agent Comparison](figs/compare_agents.png)

---

## 🗂️ Project Structure
```
TypeRL/
├── bigrams.py              # 40 tracked English bigrams
├── text_processing.py      # Bigram extraction from sentences
├── generate_dataset.py     # LLM dataset generation (Groq / Llama 3.1)
├── dataset_loader.py       # O(1) sentence sampling by (bigram, difficulty)
├── typing_env.py           # Core RL environment (MDP)
├── rule_based_agent.py     # Greedy heuristic baseline
├── q_learning.py           # Tabular Q-learning agent
├── dqn_agent.py            # Deep Q-Network agent (PyTorch)
├── compare_agents.py       # Train and compare all three agents
├── typing_component.py     # Streamlit interactive typing UI (HTML/JS)
├── typing_dataset_cleaned.csv
└── figs/                   # Output plots
```

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install numpy torch matplotlib streamlit groq python-dotenv tenacity
```

### 2. Generate the dataset (optional — skip if you have `typing_dataset_cleaned.csv`)
```bash
# Add your Groq API key to a .env file first
echo "GROQ_API_KEY=your_key_here" > .env
python generate_dataset.py
```

### 3. Train and compare agents
```bash
python compare_agents.py
```

### 4. Run individual agents
```bash
python rule_based_agent.py
python q_learning.py
python dqn_agent.py
```

---

## 🛠️ Environment Details

The environment simulates a learner's typing performance using a **logistic performance model**:

$$p_b = \sigma(k_b - d_\ell), \quad d_\ell = 0.2 \cdot \ell^{1.5}$$

For each bigram occurrence, a Bernoulli trial $X_i \sim \text{Bern}(p_b)$ is drawn, and accuracy $acc_b = \frac{1}{c_b}\sum X_i$ drives the skill update. This stochastic model reflects natural variability in human typing.

---

## 💡 Key Design Decisions

- **Logarithmic forgetting** instead of linear — prevents skill collapse during long gaps between practice
- **Nonlinear difficulty** ($\ell^{1.5}$) — better reflects real-world difficulty growth
- **$\min_b k_b$ in reward** — forces the agent to address weak bigrams instead of optimizing the average
- **LLM-generated sentences** — each (bigram, difficulty) pair has 20 natural sentences; decouples generation from the learning model
- **Full state for DQN** — unlike Q-learning, DQN sees per-bigram skills and timers, enabling fine-grained curriculum decisions

---

## 📦 Dataset

Sentences were generated using **Llama 3.1-8b (via Groq API)**.  
Each (bigram, difficulty) pair → 20 natural English sentences  
Total: `40 bigrams × 5 difficulties × 20 sentences = 4,000 sentences`

Constraints per sentence:
- Target bigram appears **≥ 5 times**
- Length: **8–12 words**
- Natural English flow, no forced repetition

---

