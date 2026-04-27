"""
TypeRL — RL Adaptive Typing Tutor
Requires:  typing_component.py  (same folder)
           typing_env.py / dqn_agent.py / text_processing.py (your files)
Run:       streamlit run app.py
"""

import streamlit as st
import numpy as np

from typing_component import typing_tutor_component
from typing_env import TypingEnv
from agents.dqn_agent import DQNAgent
from text_processing import count_tracked_bigrams, counts_to_vector

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TyperRL",
    page_icon="⌨️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=DM+Sans:wght@300;400;500;600&family=Space+Grotesk:wght@500;700&display=swap');

:root {
    --bg:      #0d0d10;
    --bg2:     #14141a;
    --bg3:     #1b1b23;
    --accent:  #e8c547;
    --acc-dim: rgba(232,197,71,0.10);
    --green:   #4ade80;
    --red:     #f87171;
    --blue:    #60a5fa;
    --text:    #f2f2f5;
    --text2:   #9999a8;
    --text3:   #4f4f5c;
    --border:  rgba(255,255,255,0.07);
    --border2: rgba(232,197,71,0.28);
    --r:       10px;
    --mono:    'JetBrains Mono', monospace;
    --ui:      'DM Sans', sans-serif;
    --disp:    'Space Grotesk', sans-serif;
}

html, body, [class*="css"] { font-family: var(--ui) !important; }
.stApp { background: var(--bg) !important; }
.block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1440px; }

#MainMenu, footer, header, .stDeployButton { display: none !important; }
.modebar { display: none !important; }
.js-plotly-plot { background: transparent !important; }

.stButton > button {
    background: transparent !important;
    border: 1px solid var(--border2) !important;
    color: var(--accent) !important;
    font-family: var(--ui) !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: .45rem 1.2rem !important;
    transition: all .18s !important;
    font-size: .88rem !important;
}
.stButton > button:hover { background: var(--acc-dim) !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: scale(.97) !important; }

[data-testid="stMetricValue"]       { font-family:var(--mono)!important; font-size:1.55rem!important; color:var(--accent)!important; }
[data-testid="stMetricLabel"]       { font-family:var(--ui)!important; color:var(--text2)!important; font-size:.7rem!important; text-transform:uppercase; letter-spacing:.09em; }
[data-testid="metric-container"]    { background:var(--bg3)!important; border:1px solid var(--border)!important; border-radius:var(--r)!important; padding:.9rem 1rem!important; }

.card { background:var(--bg3); border:1px solid var(--border); border-radius:var(--r); padding:1.1rem 1.25rem; margin-bottom:.85rem; }
.lbl  { font-size:.67rem; text-transform:uppercase; letter-spacing:.12em; color:var(--text3); margin-bottom:.45rem; font-weight:600; }

.bigram { display:inline-flex; align-items:center; background:var(--acc-dim); border:1px solid var(--border2); border-radius:7px; padding:.35rem .9rem; font-family:var(--mono); font-size:1.15rem; font-weight:600; color:var(--accent); letter-spacing:.06em; }
.pill { display:inline-flex; padding:.25rem .7rem; border-radius:5px; font-size:.67rem; font-weight:600; text-transform:uppercase; letter-spacing:.1em; }
.pill-easy   { background:rgba(74,222,128,.1);  color:#4ade80; }
.pill-medium { background:rgba(232,197,71,.1);  color:#e8c547; }
.pill-hard   { background:rgba(248,113,113,.1); color:#f87171; }

.hist-row { display:flex; justify-content:space-between; align-items:center; padding:.42rem 0; border-bottom:1px solid var(--border); font-size:.79rem; }
.hist-row:last-child { border-bottom:none; }

.br-row { display:flex; align-items:center; gap:.65rem; margin-bottom:.45rem; }
.br-lbl { font-family:var(--mono); font-size:.8rem; min-width:2.4rem; text-align:right; }
.br-bg  { flex:1; height:5px; background:var(--border); border-radius:3px; overflow:hidden; }
.br-fil { height:100%; border-radius:3px; transition:width .4s; }
.br-num { font-family:var(--mono); font-size:.7rem; color:var(--text3); min-width:2.2rem; }

::-webkit-scrollbar { width:3px; }
::-webkit-scrollbar-track { background:var(--bg2); }
::-webkit-scrollbar-thumb { background:#2e2e3a; border-radius:2px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
ss = st.session_state

def _generate_new():
    action         = ss.agent.select_action(ss.state)
    b_id, diff     = ss.env.decode_action(action)
    ss.target_bigram = ss.env.bigrams[b_id]
    ss.difficulty    = diff
    ss.sentence      = ss.env.sample_sentence(b_id, diff)
    ss.component_key = ss.get("component_key", 0) + 1

if "env" not in ss:
    ss.env   = TypingEnv()
    ss.agent = DQNAgent()
    ss.state = ss.env.reset()
    ss.skill_history = []
    ss.acc_history   = []
    ss.wpm_history   = []
    ss.completed     = 0
    ss.best_wpm      = 0.0
    ss.component_key = 0
    _generate_new()

env = ss.env


def _difficulty_pill(diff) -> str:
    d = str(diff).lower()
    if "easy" in d or d == "0":   return '<span class="pill pill-easy">Easy</span>'
    if "hard" in d or d == "2":   return '<span class="pill pill-hard">Hard</span>'
    return '<span class="pill pill-medium">Medium</span>'


def _handle_completion(result: dict):
    wpm     = float(result.get("wpm", 0))
    acc_pct = float(result.get("acc", 0))
    acc     = acc_pct / 100.0

    counts_dict = count_tracked_bigrams(ss.sentence)
    counts      = counts_to_vector(counts_dict)

    acc_vec              = np.zeros(env.K)
    acc_vec[counts > 0]  = acc

    env.update_skills(counts, acc_vec)
    env.update_timers(counts)

    ss.skill_history.append(float(np.mean(env.k)))
    ss.acc_history.append(acc)
    ss.wpm_history.append(wpm)
    ss.completed += 1
    if wpm > ss.best_wpm:
        ss.best_wpm = wpm

    ss.state = env.get_state()
    _generate_new()
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:.75rem;margin-bottom:1.5rem">
  <div style="font-family:var(--disp);font-size:1.4rem;font-weight:700;
              color:var(--accent);letter-spacing:-.02em">TypeRL</div>
  <div style="font-size:.74rem;color:var(--text3);padding:.18rem .6rem;
              background:var(--bg3);border-radius:5px;border:1px solid var(--border);
              font-family:var(--mono)">RL Adaptive</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Two-column layout
# ─────────────────────────────────────────────────────────────────────────────
main_col, side_col = st.columns([3, 1], gap="large")

with main_col:

    # Info row
    ic1, ic2, ic3, ic4 = st.columns([1, 1, 1, 1.6])
    with ic1:
        st.markdown('<div class="lbl">Target Bigram</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bigram">{ss.target_bigram}</div>', unsafe_allow_html=True)
    with ic2:
        st.markdown('<div class="lbl">Difficulty</div>', unsafe_allow_html=True)
        st.markdown(_difficulty_pill(ss.difficulty), unsafe_allow_html=True)
    with ic3:
        st.markdown('<div class="lbl">Completed</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-family:var(--mono);font-size:1.3rem;'
            f'color:var(--accent);line-height:1.8">{ss.completed}</div>',
            unsafe_allow_html=True,
        )
    with ic4:
        st.markdown("<div style='height:.05rem'></div>", unsafe_allow_html=True)
        if st.button("⟳  New Sentence", key="new_btn"):
            _generate_new()
            st.rerun()

    st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)

    # Typing component — returns dict on completion
    result = typing_tutor_component(
        sentence=ss.sentence,
        key=f"tutor_{ss.component_key}",
    )

    if result and isinstance(result, dict) and result.get("done"):
        _handle_completion(result)

    # Recent history
    if ss.acc_history:
        st.markdown("<div style='height:.85rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="lbl">Recent Rounds</div>', unsafe_allow_html=True)

        rows = ""
        for i, (a, w, s) in enumerate(zip(
            ss.acc_history[-8:][::-1],
            ss.wpm_history[-8:][::-1],
            ss.skill_history[-8:][::-1],
        )):
            idx_n  = ss.completed - i
            a_clr  = "#4ade80" if a >= .95 else ("#e8c547" if a >= .8 else "#f87171")
            rows  += (
                f'<div class="hist-row">'
                f'<span style="font-family:var(--mono);font-size:.7rem;color:var(--text3)">#{idx_n}</span>'
                f'<span style="font-family:var(--mono);color:{a_clr}">{a*100:.0f}%</span>'
                f'<span style="font-family:var(--mono)">{w:.0f} wpm</span>'
                f'<span style="font-family:var(--mono);color:var(--text3)">skill {s:.3f}</span>'
                f'</div>'
            )
        st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)


with side_col:
    import plotly.graph_objects as go

    _cfg = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        margin=dict(l=4, r=4, t=20, b=4),
        font=dict(family="DM Sans", color="#4f4f5c", size=10),
        showlegend=False,
    )
    _ax = dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.04)",
        showgrid=True,
        tickfont=dict(size=9),
        linewidth=0,
    )

    # Metrics
    st.markdown('<div class="lbl">Session Stats</div>', unsafe_allow_html=True)
    avg_acc   = float(np.mean(ss.acc_history)) if ss.acc_history else 0.0
    avg_skill = float(np.mean(env.k))

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Best WPM",  f"{ss.best_wpm:.0f}")
        st.metric("Avg Skill", f"{avg_skill:.3f}")
    with m2:
        st.metric("Avg Acc",   f"{avg_acc*100:.1f}%")
        st.metric("Total",     ss.completed)

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # Sparklines
    if ss.skill_history:
        def _spark(y, color, title, y_range=None, h=118):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=y, mode='lines+markers',
                line=dict(color=color, width=1.5, shape='spline'),
                marker=dict(color=color, size=3.5),
                fill='tozeroy',
                fillcolor=color + "18",
            ))
            layout = dict(title=dict(text=title, font=dict(size=10, color="#4f4f5c")), height=h, **_cfg)
            layout["xaxis"] = _ax
            layout["yaxis"] = dict(**_ax, **({"range": y_range} if y_range else {}))
            fig.update_layout(**layout)
            return fig

        st.plotly_chart(_spark(ss.skill_history, "#e8c547", "skill"),
                        use_container_width=True, config=dict(displayModeBar=False))
        st.plotly_chart(_spark([a*100 for a in ss.acc_history], "#4ade80", "accuracy %", [0, 105]),
                        use_container_width=True, config=dict(displayModeBar=False))
        if any(w > 0 for w in ss.wpm_history):
            st.plotly_chart(_spark(ss.wpm_history, "#60a5fa", "WPM"),
                            use_container_width=True, config=dict(displayModeBar=False))

    # Weakest bigrams
    st.markdown('<div class="lbl" style="margin-top:.5rem">Weakest Bigrams</div>', unsafe_allow_html=True)
    idx    = np.argsort(env.k)[:12]
    labels = [env.bigrams[i] for i in idx]
    values = env.k[idx]
    max_v  = max(values) if len(values) and max(values) > 0 else 1.0

    rows = ""
    for lbl, val in zip(labels, values):
        pct = (val / max_v) * 100
        clr = "#f87171" if val < 0.3 else ("#e8c547" if val < 0.6 else "#4ade80")
        rows += (
            f'<div class="br-row">'
            f'<div class="br-lbl" style="color:{clr}">{lbl}</div>'
            f'<div class="br-bg"><div class="br-fil" style="width:{pct:.0f}%;background:{clr};opacity:.6"></div></div>'
            f'<div class="br-num">{val:.2f}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)