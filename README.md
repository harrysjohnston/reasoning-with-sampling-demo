# Reasoning with Sampling – Dash demo suite

A Plotly Dash application with twelve interactive scenes illustrates why sampling from a sharpened version of a model’s own distribution can deliver stronger reasoning traces than ordinary sampling, how this differs from low temperature sampling, and how MH power sampling approaches RL-style reasoning without additional training.

Scenes

1. Distribution sharpening of a Gaussian mixture (interactive α slider)
2. Base vs RL/GRPO likelihood histograms (unit-area)
3. Local (low-temperature) token sharpening vs global sequence power sampling
4. “Few strong futures vs many weak futures” toggle
5. MH mechanics: propose/accept animation with acceptance meter
6. MH convergence intuition (reachability + randomness cues)
7. Likelihood histograms revisited: Base, Low-T, MH, RL
8. α-mass concentration curve
9. Acceptance probability heat map
10. Token sequence gallery per strategy
11. Diversity vs quality scatter
12. Strategy timeline (rolling highlighting per sampler)

Shared visual system: base model (blue), low-temperature (orange), power/MH (magenta), RL/GRPO (green), neutral accents (gray); thin strokes, rounded tokens (monospace), minimal axes, gentle transitions.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt
python app.py
```

Visit `http://127.0.0.1:8050/` to explore the interactive storyboard. The Dash dev server hot-reloads on file changes.
