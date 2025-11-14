# Reasoning with Sampling – Dash demo suite

A Plotly Dash application with eight interactive scenes illustrates why sampling from a sharpened version of a model's own distribution can deliver stronger reasoning traces than ordinary sampling, how this differs from low temperature sampling, and how MH power sampling approaches RL-style reasoning without additional training.

## Scenes

1. **Distribution sharpening** – Gaussian mixture with interactive α slider showing how p(x)^α concentrates probability mass
2. **RL-tuning sharpens the distribution** – Histogram comparing Base vs RL sequence likelihood distributions
3. **Low-temp sampling** – Next-token PDFs showing greedy-like behavior with sharpened local probabilities
4. **p^α sampling** – Next-token PDFs showing globally consistent path selection through power sampling
5. **Many weak paths vs few strong paths** – Branching graph showing how sharpening prunes weak branches
6. **MCMC frog** – Interactive Markov chain with a frog hopping between three states, showing convergence to stationary distribution
7. **Metropolis-Hastings** – Animated demonstration of the MH algorithm with deterministic accept/reject decisions showing how candidates are evaluated
8. **Likelihoods revisited** – Histogram comparing Base, MH (power sampling), and RL distributions

Shared visual system: base model (blue), low-temperature (orange), power/MH (magenta), RL/GRPO (green), neutral accents (gray); thin strokes, rounded tokens (monospace), minimal axes, gentle transitions.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt
python app.py
```

Visit `http://127.0.0.1:8050/` to explore the interactive storyboard. The Dash dev server hot-reloads on file changes.
