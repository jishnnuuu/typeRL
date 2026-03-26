"""
typing_component.py — bidirectional Streamlit ↔ JS typing component

Drop this next to app.py. In app.py, call:
    from typing_component import typing_tutor_component
    result = typing_tutor_component(sentence, key="tutor")
    if result and result.get("done"):
        # result["wpm"], result["acc"], result["elapsed"], result["errors"]
        ...
"""
import streamlit.components.v1 as components
import json

def typing_tutor_component(
    sentence: str,
    key: str = "typing_tutor",
) -> dict | None:
    """
    Renders the full interactive typing area.
    Returns a dict when the sentence is completed, else None.
    Uses Streamlit's component value passing via postMessage → declare_component.
    """
    html = _build_html(sentence)
    result = components.html(html, height=320, scrolling=False)
    return result


def _build_html(sentence: str) -> str:
    sentence_json = json.dumps(sentence)
    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ background: transparent; overflow: hidden; }}

:root {{
    --bg: #111114;
    --accent: #e8c547;
    --correct: #4ade80;
    --incorrect: #f87171;
    --pending: #3f3f46;
    --border: rgba(255,255,255,0.07);
    --border-acc: rgba(232,197,71,0.3);
    --text: #f4f4f5;
    --muted: #52525b;
    --radius: 10px;
}}

.wrap {{
    background: var(--bg);
    border: 1.5px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem 1.75rem 1.1rem;
    cursor: text;
    transition: border-color 0.25s, box-shadow 0.25s;
    position: relative;
}}
.wrap.focused {{
    border-color: var(--border-acc);
    box-shadow: 0 0 24px rgba(232,197,71,0.06);
}}
.wrap.done {{
    border-color: rgba(74,222,128,0.35);
}}

.sentence {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    line-height: 1.9;
    letter-spacing: 0.025em;
    min-height: 52px;
    user-select: none;
    word-break: break-word;
}}

.ch {{ display: inline; position: relative; }}
.ch.correct {{ color: var(--correct); }}
.ch.incorrect {{ color: var(--incorrect); border-bottom: 1.5px solid var(--incorrect); }}
.ch.pending {{ color: var(--pending); }}
.cursor-before::before {{
    content: '';
    position: absolute;
    left: -1px; top: 0.1em; bottom: 0.1em;
    width: 2.5px;
    background: var(--accent);
    border-radius: 2px;
    animation: blink 900ms step-end infinite;
}}
@keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:0}} }}

.progress-bar {{
    height: 2px;
    background: var(--border);
    margin: 0.9rem 0 0.65rem;
    border-radius: 2px;
    overflow: hidden;
}}
.progress-fill {{
    height: 100%;
    background: linear-gradient(90deg, var(--accent) 0%, #f0a500 100%);
    border-radius: 2px;
    transition: width 0.08s linear;
    width: 0%;
}}

.stats {{
    display: flex;
    gap: 2rem;
    border-top: 1px solid var(--border);
    padding-top: 0.5rem;
}}
.stat {{ display: flex; flex-direction: column; gap: 2px; }}
.stat-v {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    font-weight: 500;
    color: var(--text);
    transition: color 0.2s;
}}
.stat-l {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
}}

.hint {{
    position: absolute;
    bottom: 0.75rem;
    right: 1.25rem;
    font-size: 0.65rem;
    font-family: 'DM Sans', sans-serif;
    color: var(--muted);
    pointer-events: none;
    transition: opacity 0.3s;
}}

.result {{
    display: none;
    gap: 1.5rem;
    padding: 0.75rem 1rem;
    background: rgba(232,197,71,0.04);
    border: 1px solid rgba(232,197,71,0.25);
    border-radius: 8px;
    margin-top: 0.75rem;
}}
.result.show {{ display: flex; animation: fadeIn 0.3s; }}
.res-v {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--accent);
}}
.res-l {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
}}

.done-msg {{
    display: none;
    background: rgba(74,222,128,0.06);
    border: 1px solid rgba(74,222,128,0.22);
    border-radius: 7px;
    padding: 0.65rem 1rem;
    color: #4ade80;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    text-align: center;
    margin-top: 0.6rem;
}}
.done-msg.show {{ display: block; animation: fadeIn 0.3s; }}
kbd {{
    background: rgba(255,255,255,0.08);
    padding: 1px 6px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8em;
}}

@keyframes fadeIn {{ from{{opacity:0;transform:translateY(-3px)}} to{{opacity:1;transform:none}} }}

.hidden-in {{
    position: fixed; top:-9999px; left:-9999px;
    width:1px; height:1px; opacity:0;
}}
</style>
</head>
<body>

<div class="wrap" id="wrap" onclick="focusInput()">
    <div class="sentence" id="sent"></div>
    <div class="progress-bar"><div class="progress-fill" id="pf"></div></div>
    <div class="stats">
        <div class="stat"><div class="stat-v" id="sv-wpm">–</div><div class="stat-l">WPM</div></div>
        <div class="stat"><div class="stat-v" id="sv-acc">–</div><div class="stat-l">Accuracy</div></div>
        <div class="stat"><div class="stat-v" id="sv-t">0.0s</div><div class="stat-l">Time</div></div>
        <div class="stat"><div class="stat-v" id="sv-err">0</div><div class="stat-l">Errors</div></div>
    </div>
    <div class="hint" id="hint">Click to type</div>
</div>

<div class="result" id="result">
    <div class="stat"><div class="res-v" id="r-wpm">–</div><div class="res-l">WPM</div></div>
    <div class="stat"><div class="res-v" id="r-acc">–</div><div class="res-l">Accuracy</div></div>
    <div class="stat"><div class="res-v" id="r-t">–</div><div class="res-l">Time</div></div>
    <div class="stat"><div class="res-v" id="r-err">–</div><div class="res-l">Errors</div></div>
</div>

<div class="done-msg" id="doneMsg">
    ✓ &nbsp; Sentence complete! Press <kbd>Tab</kbd> or click <strong>New Sentence</strong> to continue.
</div>

<input class="hidden-in" type="text" id="inp"
    autocomplete="off" autocorrect="off" autocapitalize="off" spellcheck="false">

<script>
const SENTENCE = {sentence_json};
const wrap  = document.getElementById('wrap');
const sent  = document.getElementById('sent');
const pf    = document.getElementById('pf');
const hint  = document.getElementById('hint');
const inp   = document.getElementById('inp');
const res   = document.getElementById('result');
const done  = document.getElementById('doneMsg');

let typed='', t0=null, ticker=null, finished=false, errors=0, ks=0;

function render(t) {{
    let h='';
    for (let i=0;i<SENTENCE.length;i++) {{
        const ch = SENTENCE[i]===' '?'&nbsp;':SENTENCE[i];
        if (i<t.length)
            h+=`<span class="ch ${{t[i]===SENTENCE[i]?'correct':'incorrect'}}">${{ch}}</span>`;
        else if (i===t.length)
            h+=`<span class="ch pending cursor-before">${{ch}}</span>`;
        else
            h+=`<span class="ch pending">${{ch}}</span>`;
    }}
    sent.innerHTML = h;
}}

function tick() {{
     if (finished || !t0 || !ticker) return;  // ✅ strict guard

    const el = (Date.now() - t0) / 1000;
    document.getElementById('sv-t').textContent = el.toFixed(1) + 's';

    const wpm = Math.round((typed.length / 5) / (el / 60));
    const wpmEl = document.getElementById('sv-wpm');
    wpmEl.textContent = isFinite(wpm) ? wpm : '–';

    const corr = [...typed].filter((c, i) => c === SENTENCE[i]).length;
    const acc = typed.length > 0 ? Math.round(corr / typed.length * 100) : 100;
    document.getElementById('sv-acc').textContent = acc + '%';
}}

function onInput() {{
    if (finished) return;  // stop everything after completion

    const v = inp.value;

    // ⏱️ Start timer on first keystroke
    if (!t0 && v.length > 0) {{
        t0 = Date.now();
        hint.style.opacity = '0';
        ticker = setInterval(tick, 80);
    }}

    // 🧠 Count keystrokes + errors
    if (v.length > typed.length) {{
        const i = v.length - 1;
        ks++;

        // only compare within sentence bounds
        if (i < SENTENCE.length && v[i] !== SENTENCE[i]) {{
            errors++;
        }}

        document.getElementById('sv-err').textContent = errors;
    }}

    // ✅ KEEP RAW INPUT (important)
    typed = v;

    // 👀 Render only up to sentence length
    const displayText = typed.slice(0, SENTENCE.length);
    render(displayText);

    // 📊 Progress bar (clamped visually)
    const progress = Math.min(typed.length, SENTENCE.length) / SENTENCE.length;
    pf.style.width = (progress * 100) + '%';

    // ✅ ROBUST COMPLETION (handles extra typing)
    if (typed.length >= SENTENCE.length) {{
        typed = typed.slice(0, SENTENCE.length);  // final clamp
        inp.value = typed;
        finish();
    }}
}}

function finish() {{
    if (finished) return;   // prevent double execution
    finished = true;

    // ⛔ stop timer completely
    clearInterval(ticker);
    ticker = null;

    const el = (Date.now() - t0) / 1000;

    // 📊 metrics
    const wpm = Math.round((SENTENCE.length / 5) / (el / 60));
    const corr = [...typed].filter((c, i) => c === SENTENCE[i]).length;
    const acc = Math.round(corr / SENTENCE.length * 100);

    // 🧊 freeze UI values
    document.getElementById('sv-t').textContent = el.toFixed(1) + 's';
    document.getElementById('sv-wpm').textContent = wpm;
    document.getElementById('sv-acc').textContent = acc + '%';

    // 🎨 UI state
    wrap.classList.remove('focused');
    wrap.classList.add('done');

    res.classList.add('show');
    done.classList.add('show');

    // 📦 result panel
    document.getElementById('r-wpm').textContent = wpm;
    document.getElementById('r-acc').textContent = acc + '%';
    document.getElementById('r-t').textContent = el.toFixed(2) + 's';
    document.getElementById('r-err').textContent = errors;

    // 🔒 disable further typing
    inp.blur();
    inp.disabled = true;

    // 📤 send result (only once)
    window.parent.postMessage({{
        isStreamlitMessage: true,
        type: "streamlit:componentValue",
        value: {{done: true, wpm, acc, elapsed: el, errors }}
    }}, '*');
}}

function focusInput() {{
    inp.focus();
    if (!finished) wrap.classList.add('focused');
}}
inp.addEventListener('input', onInput);
inp.addEventListener('blur', ()=>{{if(!finished)wrap.classList.remove('focused');}});
inp.addEventListener('keydown', e=>{{if(e.key==='Tab')e.preventDefault();}});

render('');
</script>
</body>
</html>
"""