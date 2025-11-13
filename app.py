"""Sharpened sampling demos: Reasoning with Sampling

This Dash app visualizes how exponentiating likelihoods (p(x)^alpha) affects
sampling behavior and how this differs from classic low-temperature sampling
and RL-tuned models. It presents five small, self-contained scenes:

- Scene 01 — Distribution sharpening:
  Plots a 1D mixture density p(x) alongside its sharpened version p(x)^alpha,
  both normalized to unit area. Use the alpha slider to increase emphasis
  on high-likelihood regions.

- Scene 02 — RL-tuning sharpens the distribution:
  Histogram of average log-likelihoods comparing Base vs RL. RL concentrates
  mass at higher average likelihoods.

- Scene 03 — Low-temp sampling:
  Next-token PDFs for a small vocabulary at each step. The “low-temp” PDF
  is a sharpened version of the base PDF; the selected token follows an
  argmax-like, most-probable (greedy) path, building the familiar sentence.

- Scene 04 — p^alpha sampling:
  Next-token PDFs where the overlay favors a different, globally consistent
  path. The selected tokens compose a rarer, more valuable path.

- Scene 05 — Likelihoods revisited:
  Histogram comparing Base, MH-style power sampling, and RL. MH samples
  from a sharpened distribution, similar to RL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from dash import Dash, Input, Output, State, dcc, html
import plotly.graph_objects as go

app = Dash(__name__,
    external_scripts=["https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"],)
server = app.server

palette = {
    "base": "#2563eb",
    "low": "#f97316",
    "power": "#d946ef",
    "rl": "#22c55e",
    "neutral": "#94a3b8",
    "ink": "#0f172a",
    "axis": "rgba(15, 23, 42, 0.18)",
}

# ---------- Scene 01: Distribution sharpening ----------

mix_components = (
    {"weight": 0.55, "mean": -1.2, "variance": 0.9},
    {"weight": 0.45, "mean": 1.7, "variance": 0.4},
)

def mixture_pdf(x: float) -> float:
    value = 0.0
    for comp in mix_components:
        denom = math.sqrt(2 * math.pi * comp["variance"])
        exponent = -((x - comp["mean"]) ** 2) / (2 * comp["variance"])
        value += comp["weight"] * math.exp(exponent) / denom
    return value

x_values = np.linspace(-4, 4, 1000)


def distribution_figure(alpha: float) -> go.Figure:
    base = np.array([mixture_pdf(x) for x in x_values])
    base_area = np.trapezoid(base, x_values)
    base_scaled = base / base_area
    sharpened = np.power(base, alpha)
    sharpened_area = np.trapezoid(sharpened, x_values)
    sharpened_scaled = sharpened / sharpened_area

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=base_scaled,
            mode="lines",
            name=r"$p(x)$",
            line=dict(color=palette["base"], width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=sharpened_scaled,
            mode="lines",
            name=r"$p(x)^{\alpha}$",
            line=dict(color=palette["power"], width=3),
        )
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        template="simple_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=palette["axis"]),
        yaxis=dict(range=[0, 0.8], showgrid=False, zeroline=False, linecolor=palette["axis"]),
    )
    return fig


# ---------- Likelihood data for histograms ----------

rng = np.random.default_rng(42)

LIKELIHOOD_MIN = -0.5
LIKELIHOOD_MAX = -1e-3
LIKELIHOOD_MAX_SAFE = np.nextafter(LIKELIHOOD_MAX, LIKELIHOOD_MIN)


def truncated_normal_sample(
    mean: float,
    std: float,
    size: int,
    lower: float,
    upper: float,
    oversample_factor: int = 3,
) -> np.ndarray:
    """Draw samples from a normal distribution confined to [lower, upper]."""
    collected: List[np.ndarray] = []
    remaining = size
    while remaining > 0:
        batch_size = max(remaining * oversample_factor, remaining)
        batch = rng.normal(mean, std, batch_size)
        accepted = batch[(batch >= lower) & (batch <= upper)]
        if accepted.size == 0:
            continue
        take = min(remaining, accepted.size)
        collected.append(accepted[:take])
        remaining -= take
    return np.concatenate(collected)


def sample_likelihood(mean: float, std: float, size: int) -> np.ndarray:
    return truncated_normal_sample(
        mean=mean,
        std=std,
        size=size,
        lower=LIKELIHOOD_MIN,
        upper=LIKELIHOOD_MAX_SAFE,
    )


likelihood_sets = {
    "base": sample_likelihood(-0.1, 0.1, 10000),
    "low": sample_likelihood(-0.08, 0.05, 10000),
    "mh": sample_likelihood(-0.08, 0.04, 10000),
    "rl": sample_likelihood(-0.06, 0.03, 10000),
}


people_labels = {
    "base": "Base",
    "low": "Low-temp",
    "mh": "MH (power)",
    "rl": "RL/GRPO",
}


color_lookup = {
    "base": palette["base"],
    "low": palette["low"],
    "mh": palette["power"],
    "rl": palette["rl"],
}


def likelihood_histogram(keys: Sequence[str]) -> go.Figure:
    fig = go.Figure()
    bin_start = LIKELIHOOD_MIN
    bin_end = LIKELIHOOD_MAX
    num_bins = 40
    bin_edges = np.linspace(bin_start, bin_end, num_bins + 1)
    for key in keys:
        values = likelihood_sets[key]
        hist, edges = np.histogram(values, bins=bin_edges, density=True)
        step_x = np.repeat(edges, 2)
        step_y = np.concatenate(([0.0], np.repeat(hist, 2), [0.0]))
        fig.add_trace(
            go.Scatter(
                x=step_x,
                y=step_y,
                name=people_labels[key],
                mode="lines",
                line=dict(color=color_lookup[key], width=2),
                hovertemplate="average log-likelihood=%{x:.3f}<br>density=%{y:.3f}<extra>" + people_labels[key] + "</extra>",
            )
        )
    fig.update_layout(
        barmode="overlay",
        template="simple_white",
        margin=dict(l=20, r=20, t=10, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="average log-likelihood", range=[LIKELIHOOD_MIN, 0], showgrid=False, linecolor=palette["axis"]),
        yaxis=dict(title="density", showgrid=False, linecolor=palette["axis"]),
    )
    return fig


# ---------- Scene 03: deterministic next-token pdfs ----------

BASE_PREFIX = ["The", "quick"]


def normalize(values: Sequence[float]) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return arr
    arr[arr < 0] = 0.0
    total = arr.sum()
    if total <= 0:
        return np.full(arr.shape, 1.0 / arr.size)
    return arr / total


# Brown path: always choose first (highest probability)
BROWN_FRAMES = [
    {
        "tokens": ["brown", "answer", "response", "fix", "sand"],
        "base_probs": (base := normalize([0.32, 0.22, 0.20, 0.15, 0.11])),
        "overlay_probs": normalize(np.power(base, 2)),
        "selected": "brown",
    },
    {
        "tokens": ["fox", "hare", "mouse", "rabbit", "squirrel"],
        "base_probs": (base := normalize([0.28, 0.25, 0.20, 0.15, 0.12])),
        "overlay_probs": normalize(np.power(base, 2)),
        "selected": "fox",
    },
    {
        "tokens": ["jumps", "runs", "leaps", "bounds", "shoots"],
        "base_probs": (base := normalize([0.30, 0.25, 0.20, 0.15, 0.10])),
        "overlay_probs": normalize(np.power(base, 2)),
        "selected": "jumps",
    },
    {
        "tokens": ["over", "through", "past", "around", "under"],
        "base_probs": (base := normalize([0.28, 0.24, 0.22, 0.16, 0.10])),
        "overlay_probs": normalize(np.power(base, 2)),
        "selected": "over",
    },
    {
        "tokens": ["the lazy dog", "the tall grass", "the fallen log", "the quiet field", "the wooden fence"],
        "base_probs": (base := normalize([0.26, 0.24, 0.22, 0.18, 0.10])),
        "overlay_probs": normalize(np.power(base, 2)),
        "selected": "the lazy dog",
    },
]

# Sand path: choose marked options (!)
SAND_FRAMES = [
    {
        "tokens": ["brown", "answer", "response", "fix", "sand"],
        "base_probs": normalize([0.32, 0.22, 0.20, 0.15, 0.11]),
        "overlay_probs": normalize([0.15, 0.12, 0.10, 0.13, 0.50]),
        "selected": "sand",
    },
    {
        "tokens": ["pulls", "shifts", "quivers", "devours", "consumes"],
        "base_probs": normalize([0.25, 0.24, 0.22, 0.18, 0.11]),
        "overlay_probs": normalize([0.12, 0.15, 0.50, 0.13, 0.10]),
        "selected": "quivers",
    },
    {
        "tokens": ["with", "as", "after", "when", "uncontrollably"],
        "base_probs": normalize([0.28, 0.26, 0.22, 0.14, 0.10]),
        "overlay_probs": normalize([0.13, 0.50, 0.15, 0.12, 0.10]),
        "selected": "as",
    },
    {
        "tokens": ["the dunes", "its victim", "we pass", "my senses", "if with"],
        "base_probs": normalize([0.30, 0.24, 0.20, 0.16, 0.10]),
        "overlay_probs": normalize([0.12, 0.50, 0.15, 0.13, 0.10]),
        "selected": "its victim",
    },
    {
        "tokens": ["ceases to struggle", "thrashes within", "succumbs to fatigue", "loses hope", "curses the gods"],
        "base_probs": normalize([0.26, 0.24, 0.22, 0.18, 0.10]),
        "overlay_probs": normalize([0.12, 0.13, 0.50, 0.15, 0.10]),
        "selected": "succumbs to fatigue",
    },
]

SCENE_SEQUENCE_LENGTH = len(BROWN_FRAMES)


def get_frame_data(mode: str, frame: int) -> dict:
    """Get frame data for the given mode and frame index."""
    frames = BROWN_FRAMES if mode == "low" else SAND_FRAMES
    if 0 <= frame < len(frames):
        return frames[frame]
    return frames[0]


def scene_step(store: dict | None, mode: str) -> dict:
    prev_frame = int((store or {}).get("frame", -1))
    frame = (prev_frame + 1) % SCENE_SEQUENCE_LENGTH
    suffix = [] if frame == 0 else list((store or {}).get("suffix", []))
    
    frame_data = get_frame_data(mode, frame)
    selected = frame_data["selected"]
    
    if selected:
        suffix.append(selected)
    
    return {
        "frame": frame,
        "suffix": suffix,
        "tokens": frame_data["tokens"],
        "base_probs": frame_data["base_probs"].tolist(),
        "overlay_probs": frame_data["overlay_probs"].tolist(),
        "selected": selected,
    }


INITIAL_BASE_SCENE3 = scene_step(None, "low")
INITIAL_FULL_SCENE3 = scene_step(None, "full")


def scene3_prefix_spans(suffix: Sequence[str], variant: str) -> List[html.Span]:
    spans: List[html.Span] = []
    for idx, word in enumerate(BASE_PREFIX):
        spans.append(html.Span(word, className="token token-neutral stem-token", key=f"stem-{idx}"))
    for idx, token in enumerate(suffix):
        spans.append(
            html.Span(
                token,
                className="token token-highlight",
                key=f"{variant}-selected-{idx}",
                **{"data-variant": variant},
            )
        )
    return spans


def scene3_figure(
    tokens: Sequence[str],
    base_probs: Sequence[float],
    overlay_probs: Sequence[float],
    selected_token: str | None,
    overlay_name: str,
    overlay_color: str,
) -> go.Figure:
    tokens = list(tokens)
    base = np.array(base_probs, dtype=float)
    overlay = np.array(overlay_probs, dtype=float)
    if not tokens or base.size == 0 or overlay.size == 0 or len(tokens) != base.size or len(tokens) != overlay.size:
        fig = go.Figure()
        fig.add_annotation(
            text="Waiting for sequence...",
            showarrow=False,
            font=dict(color=palette["neutral"], size=14),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        fig.update_layout(
            template="simple_white",
            margin=dict(l=20, r=20, t=10, b=40),
            xaxis=dict(title="candidate next token", showgrid=False, linecolor=palette["axis"]),
            yaxis=dict(title="probability", showgrid=False, linecolor=palette["axis"]),
        )
        return fig

    highlight_color = palette["rl"]
    highlight_base = [highlight_color if token == selected_token else "rgba(15, 23, 42, 0.18)" for token in tokens]
    highlight_overlay = [highlight_color if token == selected_token else "rgba(15, 23, 42, 0.15)" for token in tokens]
    line_widths = [3 if token == selected_token else 0.8 for token in tokens]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=tokens,
            y=base,
            name="Base pdf",
            marker=dict(color=palette["base"], line=dict(color=highlight_base, width=line_widths)),
            width=0.38,
            offsetgroup="base",
            hovertemplate="%{x}<br>p=%{y:.2f}<extra>Base</extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=tokens,
            y=overlay,
            name=overlay_name,
            marker=dict(color=overlay_color, line=dict(color=highlight_overlay, width=line_widths)),
            width=0.38,
            offsetgroup="overlay",
            hovertemplate="%{x}<br>p=%{y:.2f}<extra>" + overlay_name + "</extra>",
        )
    )
    fig.update_layout(
        barmode="group",
        template="simple_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=10, b=40),
        xaxis=dict(title="candidate next token", showgrid=False, linecolor=palette["axis"]),
        yaxis=dict(title="probability", showgrid=False, linecolor=palette["axis"]),
    )
    return fig


# ---------- Layout helpers ----------
def card(identifier: str, scene: str, title: str, caption: str | None, children: Sequence[object]) -> html.Section:
    return html.Section(
        className="demo",
        id=identifier,
        children=[
            html.Div(
                className="demo-header",
                children=[
                    html.Div(
                        children=[
                            html.P(scene, className="eyebrow"),
                            dcc.Markdown(title, className="card-title", mathjax=True),
                        ]
                    ),
                    dcc.Markdown(caption, className="caption", mathjax=True) if caption else None,
                ],
            ),
            *children,
        ],
    )


# ---------- App layout ----------

app.layout = html.Div(
    children=[
        html.Header(
            className="page-header",
            children=[
                html.Div(
                    children=[
                        html.P("Reasoning with Sampling", className="eyebrow"),
                        html.H1("Sharpened sampling demos"),
                        html.P(
                            "Five scenes illustrate how sampling from a sharpened distribution differs from low temperature, "
                            "and how MH power sampling approaches RL-style reasoning without training.",
                            className="lede",
                        ),
                    ]
                )
            ],
        ),
        html.Main(
            className="page-main",
            children=[
                card(
                    "scene-distribution",
                    "Scene 01",
                    "Distribution sharpening",
                    None,
                    [
                        dcc.Graph(id="distribution-graph", config={"displayModeBar": False}),
                        html.Div(
                            className="inline-control alpha-control",
                            children=[
                                dcc.Markdown("$\\alpha$", className="alpha-symbol", mathjax=True),
                                dcc.Slider(
                                    id="alpha-slider",
                                    min=1,
                                    max=4,
                                    step=0.1,
                                    value=4.0,
                                    marks={i: str(i) for i in range(1, 5)},
                                    tooltip={"placement": "bottom", "always_visible": False},
                                    updatemode="drag",
                                    className="alpha-slider",
                                ),
                                html.Span("4.0", id="alpha-label", className="value"),
                            ],
                        ),
                    ],
                ),
                card(
                    "scene-rl-hist",
                    "Scene 02",
                    "RL-tuning sharpens the distribution",
                    None,
                    [dcc.Graph(id="rl-hist-graph", config={"displayModeBar": False})],
                ),
                card(
                    "scene-low-temp",
                    "Scene 03",
                    "Low-temp sampling",
                    None,
                    [
                        dcc.Graph(id="base-next-graph", config={"displayModeBar": False}),
                        html.Div(id="base-prefix-line", className="prefix-line token-row"),
                    ],
                ),
                card(
                    "scene-global-power",
                    "Scene 04",
                    "$p^{\\alpha}$ sampling",
                    None,
                    [
                        dcc.Graph(id="low-temp-next-graph", config={"displayModeBar": False}),
                        html.Div(id="low-temp-prefix-line", className="prefix-line token-row"),
                    ],
                ),
                card(
                    "scene-hist-mh",
                    "Scene 05",
                    "Likelihoods revisited",
                    None,
                    [dcc.Graph(id="mh-hist-graph", config={"displayModeBar": False})],
                ),
            ],
        ),
        html.Footer(children=[html.P("Palette: base (blue), low-temp (orange), power/MH (magenta), RL (green), neutral (gray).", className="inline-note")]),
        dcc.Interval(id="scene3-interval", interval=3200, n_intervals=0),
        dcc.Store(id="base-prefix-store", data=INITIAL_BASE_SCENE3),
        dcc.Store(id="low-prefix-store", data=INITIAL_FULL_SCENE3),
    ]
)


# ---------- Callbacks ----------

@app.callback(Output("distribution-graph", "figure"), Output("alpha-label", "children"), Input("alpha-slider", "value"))
def update_distribution(alpha: float):
    return distribution_figure(alpha), f"{alpha:.1f}"


@app.callback(Output("rl-hist-graph", "figure"), Input("alpha-slider", "value"))
def init_rl_hist(_):
    return likelihood_histogram(["base", "rl"])


@app.callback(Output("mh-hist-graph", "figure"), Input("alpha-slider", "value"))
def init_mh_hist(_):
    return likelihood_histogram(["base", "mh", "rl"])


@app.callback(
    Output("base-prefix-store", "data"),
    Output("low-prefix-store", "data"),
    Input("scene3-interval", "n_intervals"),
    State("base-prefix-store", "data"),
    State("low-prefix-store", "data"),
    State("alpha-slider", "value"),
)
def advance_prefixes(n_intervals, base_store, low_store, _alpha):
    if n_intervals is None or n_intervals == 0:
        return base_store, low_store
    return scene_step(base_store, "low"), scene_step(low_store, "full")


@app.callback(
    Output("base-next-graph", "figure"),
    Output("base-prefix-line", "children"),
    Input("base-prefix-store", "data"),
    Input("alpha-slider", "value"),
)
def render_base_next(store, _alpha):
    data = store or scene_step(None, "low")
    tokens = data.get("tokens", [])
    base_probs = np.array(data.get("base_probs", []), dtype=float)
    overlay = np.array(data.get("overlay_probs", []), dtype=float)
    figure = scene3_figure(tokens, base_probs, overlay, data.get("selected"), "Low-temp", palette["low"])
    prefix = scene3_prefix_spans(data.get("suffix", []), variant="base")
    return figure, prefix


@app.callback(
    Output("low-temp-next-graph", "figure"),
    Output("low-temp-prefix-line", "children"),
    Input("low-prefix-store", "data"),
    Input("alpha-slider", "value"),
)
def render_low_temp_next(store, _alpha):
    data = store or scene_step(None, "full")
    tokens = data.get("tokens", [])
    base_probs = np.array(data.get("base_probs", []), dtype=float)
    overlay = np.array(data.get("overlay_probs", []), dtype=float)
    figure = scene3_figure(tokens, base_probs, overlay, data.get("selected"), "$p^{\\alpha}$", palette["power"])
    prefix = scene3_prefix_spans(data.get("suffix", []), variant="power")
    return figure, prefix


if __name__ == "__main__":
    app.run(debug=True)
