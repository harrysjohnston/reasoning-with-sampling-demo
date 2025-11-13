"""Sharpened sampling demos: Reasoning with Sampling

This Dash app visualizes how exponentiating likelihoods (p(x)^alpha) affects
sampling behavior and how this differs from classic low-temperature sampling
and RL-tuned models. It presents six small, self-contained scenes:

- Scene 01 — Distribution sharpening:
  Plots a 1D mixture density p(x) alongside its sharpened version p(x)^alpha,
  both normalized to unit area. Use the alpha slider to increase emphasis
  on high-likelihood regions.

- Scene 02 — RL-tuning sharpens the distribution:
  Histogram of average log-likelihoods comparing Base vs RL. RL concentrates
  mass at higher average likelihoods.

- Scene 03 — Low-temp sampling:
  Next-token PDFs for a small vocabulary at each step. The "low-temp" PDF
  is a sharpened version of the base PDF; the selected token follows an
  argmax-like, most-probable (greedy) path, building the familiar sentence.

- Scene 04 — p^alpha sampling:
  Next-token PDFs where the overlay favors a different, globally consistent
  path. The selected tokens compose a rarer, more valuable path.

- Scene 05 — Paths of sand vs brown:
  Network graph showing branching differences between "brown" (many weak paths)
  and "sand" (few strong paths). Sharpening prunes weak branches.

- Scene 06 — Likelihoods revisited:
  Histogram comparing Base, MH-style power sampling, and RL. MH samples
  from a sharpened distribution, similar to RL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import dash
from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import dash_cytoscape as cyto

app = Dash(__name__,
    external_scripts=["https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"],)
server = app.server

# Light mode palette
palette_light = {
    "base": "#2563eb",
    "low": "#f97316",
    "power": "#d946ef",
    "rl": "#22c55e",
    "neutral": "#94a3b8",
    "ink": "#0f172a",
    "axis": "rgba(15, 23, 42, 0.18)",
    "bg": "#ffffff",
    "grid": "rgba(15, 23, 42, 0.05)",
}

# Dark mode palette
palette_dark = {
    "base": "#60a5fa",
    "low": "#fb923c",
    "power": "#e879f9",
    "rl": "#4ade80",
    "neutral": "#64748b",
    "ink": "#f1f5f9",
    "axis": "rgba(241, 245, 249, 0.18)",
    "bg": "#1e293b",
    "grid": "rgba(241, 245, 249, 0.05)",
}

# Default to light palette
palette = palette_light

def get_palette(theme: str | None) -> dict:
    """Get the appropriate color palette based on theme."""
    if theme == "dark":
        return palette_dark
    return palette_light

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


def distribution_figure(alpha: float, theme: str | None = None) -> go.Figure:
    pal = get_palette(theme)
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
            line=dict(color=pal["base"], width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=sharpened_scaled,
            mode="lines",
            name=r"$p(x)^{\alpha}$",
            line=dict(color=pal["power"], width=3),
        )
    )
    yaxis_config = dict(showgrid=False, zeroline=False, linecolor=pal["axis"], tickcolor=pal["ink"], tickfont=dict(color=pal["ink"]))
    if alpha < 5:
        yaxis_config["range"] = [0, 0.8]
    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        template=None,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=pal["ink"])),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=pal["axis"], tickcolor=pal["ink"], tickfont=dict(color=pal["ink"])),
        yaxis=yaxis_config,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=pal["ink"]),
        hoverlabel=dict(bgcolor=pal["bg"], font=dict(color=pal["ink"]), bordercolor=pal["axis"]),
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


def likelihood_histogram(keys: Sequence[str], theme: str | None = None) -> go.Figure:
    pal = get_palette(theme)
    # Update color lookup for current theme
    color_lookup_themed = {
        "base": pal["base"],
        "low": pal["low"],
        "mh": pal["power"],
        "rl": pal["rl"],
    }
    
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
                line=dict(color=color_lookup_themed[key], width=2),
                hovertemplate="average log-likelihood=%{x:.3f}<br>density=%{y:.3f}<extra>" + people_labels[key] + "</extra>",
            )
        )
    fig.update_layout(
        barmode="overlay",
        template=None,
        margin=dict(l=20, r=20, t=10, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=pal["ink"])),
        xaxis=dict(title="sequence log-likelihood / length", range=[LIKELIHOOD_MIN, 0], showgrid=False, linecolor=pal["axis"], tickcolor=pal["ink"], tickfont=dict(color=pal["ink"])),
        yaxis=dict(title="density", showgrid=False, linecolor=pal["axis"], tickcolor=pal["ink"], tickfont=dict(color=pal["ink"])),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=pal["ink"]),
        hoverlabel=dict(bgcolor=pal["bg"], font=dict(color=pal["ink"]), bordercolor=pal["axis"]),
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
alpha = 4.0
BROWN_FRAMES = [
    {
        "tokens": ["brown", "answer", "response", "fix", "sand"],
        "base_probs": (base := normalize([0.32, 0.22, 0.20, 0.15, 0.11])),
        "overlay_probs": normalize(np.power(base, alpha)),
        "selected": "brown",
    },
    {
        "tokens": ["fox", "hare", "mouse", "rabbit", "squirrel"],
        "base_probs": (base := normalize([0.28, 0.25, 0.20, 0.15, 0.12])),
        "overlay_probs": normalize(np.power(base, alpha)),
        "selected": "fox",
    },
    {
        "tokens": ["jumps", "runs", "leaps", "bounds", "shoots"],
        "base_probs": (base := normalize([0.30, 0.25, 0.20, 0.15, 0.10])),
        "overlay_probs": normalize(np.power(base, alpha)),
        "selected": "jumps",
    },
    {
        "tokens": ["over", "through", "past", "around", "under"],
        "base_probs": (base := normalize([0.28, 0.24, 0.22, 0.16, 0.10])),
        "overlay_probs": normalize(np.power(base, alpha)),
        "selected": "over",
    },
    {
        "tokens": ["the lazy dog", "the tall grass", "the fallen log", "the quiet field", "the wooden fence"],
        "base_probs": (base := normalize([0.26, 0.24, 0.22, 0.18, 0.10])),
        "overlay_probs": normalize(np.power(base, alpha)),
        "selected": "the lazy dog",
    },
]

# Sand path: choose marked options (!)
path_prob = 0.35
SAND_FRAMES = [
    {
        "tokens": ["brown", "answer", "response", "fix", "sand"],
        "base_probs": normalize([0.32, 0.22, 0.20, 0.15, 0.11]),
        "overlay_probs": normalize([0.15, 0.12, 0.10, 0.13, path_prob]),
        "selected": "sand",
    },
    {
        "tokens": ["pulls", "shifts", "quivers", "devours", "consumes"],
        "base_probs": normalize([0.25, 0.24, 0.22, 0.18, 0.11]),
        "overlay_probs": normalize([0.12, 0.15, path_prob, 0.13, 0.10]),
        "selected": "quivers",
    },
    {
        "tokens": ["with", "as", "after", "when", "uncontrollably"],
        "base_probs": normalize([0.28, 0.26, 0.22, 0.14, 0.10]),
        "overlay_probs": normalize([0.13, path_prob, 0.15, 0.12, 0.10]),
        "selected": "as",
    },
    {
        "tokens": ["the dunes", "its victim", "we pass", "my senses", "if with"],
        "base_probs": normalize([0.30, 0.24, 0.20, 0.16, 0.10]),
        "overlay_probs": normalize([0.12, path_prob, 0.15, 0.13, 0.10]),
        "selected": "its victim",
    },
    {
        "tokens": ["ceases to struggle", "thrashes within", "succumbs to fatigue", "loses hope", "curses the gods"],
        "base_probs": normalize([0.26, 0.24, 0.22, 0.18, 0.10]),
        "overlay_probs": normalize([0.12, 0.13, path_prob, 0.15, 0.10]),
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
    theme: str | None = None,
) -> go.Figure:
    pal = get_palette(theme)
    tokens = list(tokens)
    base = np.array(base_probs, dtype=float)
    overlay = np.array(overlay_probs, dtype=float)
    if not tokens or base.size == 0 or overlay.size == 0 or len(tokens) != base.size or len(tokens) != overlay.size:
        fig = go.Figure()
        fig.add_annotation(
            text="Waiting for sequence...",
            showarrow=False,
            font=dict(color=pal["neutral"], size=14),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        fig.update_layout(
            template=None,
            margin=dict(l=20, r=20, t=10, b=40),
            xaxis=dict(title="candidate next token", showgrid=False, linecolor=pal["axis"], tickcolor=pal["ink"], tickfont=dict(color=pal["ink"])),
            yaxis=dict(title="probability", showgrid=False, linecolor=pal["axis"], tickcolor=pal["ink"], tickfont=dict(color=pal["ink"])),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=pal["ink"]),
        )
        return fig

    highlight_color = pal["rl"]
    highlight_base = [highlight_color if token == selected_token else pal["axis"] for token in tokens]
    highlight_overlay = [highlight_color if token == selected_token else pal["axis"] for token in tokens]
    line_widths = [3 if token == selected_token else 0.8 for token in tokens]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=tokens,
            y=base,
            name="Base pdf",
            marker=dict(color=pal["base"], line=dict(color=highlight_base, width=line_widths)),
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
        template=None,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=pal["ink"])),
        margin=dict(l=20, r=20, t=10, b=40),
        xaxis=dict(title="candidate next token", showgrid=False, linecolor=pal["axis"], tickcolor=pal["ink"], tickfont=dict(color=pal["ink"])),
        yaxis=dict(title="probability", showgrid=False, linecolor=pal["axis"], tickcolor=pal["ink"], tickfont=dict(color=pal["ink"])),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=pal["ink"]),
        hoverlabel=dict(bgcolor=pal["bg"], font=dict(color=pal["ink"]), bordercolor=pal["axis"]),
    )
    return fig


# ---------- Scene 05: Branching structure (brown vs sand) ----------

# Graph data structure: nodes and edges with base weights
BRANCH_NODES = [
    {"id": "root", "label": "the quick", "group": "neutral"},
    {"id": "brown", "label": "brown", "group": "brown"},
    {"id": "sand", "label": "sand", "group": "sand"},
    # Brown path: many animals
    {"id": "fox", "label": "fox", "group": "brown"},
    {"id": "rabbit", "label": "rabbit", "group": "brown"},
    {"id": "hare", "label": "hare", "group": "brown"},
    {"id": "squirrel", "label": "squirrel", "group": "brown"},
    {"id": "mouse", "label": "mouse", "group": "brown"},
    # Brown downstream (hinted, many weak actions)
    {"id": "runs", "label": "runs", "group": "brown"},
    {"id": "leaps", "label": "leaps", "group": "brown"},
    {"id": "bounds", "label": "bounds", "group": "brown"},
    {"id": "jumps", "label": "jumps", "group": "brown"},
    {"id": "dashes", "label": "dashes", "group": "brown"},
    # Sand path: few verbs
    {"id": "quivers", "label": "quivers", "group": "sand"},
    {"id": "consumes", "label": "consumes", "group": "sand"},
    {"id": "shifts", "label": "shifts", "group": "sand"},
]

BRANCH_EDGES = [
    # Root to choice nodes
    {"source": "root", "target": "brown", "weight": 0.55},
    {"source": "root", "target": "sand", "weight": 0.45},
    # Brown to animals (spread thin, ~0.18 each)
    {"source": "brown", "target": "fox", "weight": 0.22},
    {"source": "brown", "target": "rabbit", "weight": 0.20},
    {"source": "brown", "target": "hare", "weight": 0.18},
    {"source": "brown", "target": "squirrel", "weight": 0.18},
    {"source": "brown", "target": "mouse", "weight": 0.16},
    # Each animal to many actions (very thin, 0.03-0.06)
    {"source": "fox", "target": "runs", "weight": 0.06},
    {"source": "fox", "target": "leaps", "weight": 0.05},
    {"source": "fox", "target": "jumps", "weight": 0.04},
    {"source": "fox", "target": "dashes", "weight": 0.03},
    {"source": "rabbit", "target": "runs", "weight": 0.05},
    {"source": "rabbit", "target": "bounds", "weight": 0.06},
    {"source": "rabbit", "target": "leaps", "weight": 0.04},
    {"source": "rabbit", "target": "jumps", "weight": 0.03},
    {"source": "hare", "target": "dashes", "weight": 0.06},
    {"source": "hare", "target": "runs", "weight": 0.05},
    {"source": "hare", "target": "leaps", "weight": 0.04},
    {"source": "squirrel", "target": "jumps", "weight": 0.06},
    {"source": "squirrel", "target": "bounds", "weight": 0.05},
    {"source": "squirrel", "target": "leaps", "weight": 0.03},
    {"source": "mouse", "target": "runs", "weight": 0.05},
    {"source": "mouse", "target": "dashes", "weight": 0.04},
    # Sand to verbs (few, strong: ~0.30 each)
    {"source": "sand", "target": "quivers", "weight": 0.35},
    {"source": "sand", "target": "consumes", "weight": 0.30},
    {"source": "sand", "target": "shifts", "weight": 0.28},
]


def compute_threshold(alpha: float) -> float:
    """Compute visibility threshold based on alpha."""
    # Use a very low threshold - rely on opacity for visibility control
    return np.interp(alpha, [1.0, 4.0, 10.0], [0.0001, 0.0005, 0.002])


def build_branch_elements(alpha: float, theme: str | None = None):
    """
    Build Cytoscape elements and stylesheet for the branching graph.
    
    Args:
        alpha: Sharpening parameter (1.0 to 10.0)
        theme: Color theme ('light' or 'dark')
    
    Returns:
        (elements, stylesheet) tuple
    """
    pal = get_palette(theme)
    threshold = compute_threshold(alpha)
    
    # Compute effective edge strengths
    edges_with_strength = []
    for edge in BRANCH_EDGES:
        base_weight = edge["weight"]
        effective_strength = base_weight ** alpha
        if effective_strength >= threshold:
            edges_with_strength.append({
                "source": edge["source"],
                "target": edge["target"],
                "strength": effective_strength,
            })
    
    # Build set of visible nodes (nodes with visible edges + root/choice nodes)
    visible_nodes = {"root", "brown", "sand"}
    for edge in edges_with_strength:
        visible_nodes.add(edge["source"])
        visible_nodes.add(edge["target"])
    
    # Compute max incoming edge strength for each node (for opacity)
    node_max_incoming = {}
    for edge in edges_with_strength:
        target = edge["target"]
        strength = edge["strength"]
        if target not in node_max_incoming:
            node_max_incoming[target] = strength
        else:
            node_max_incoming[target] = max(node_max_incoming[target], strength)
    
    # Build node elements
    node_elements = []
    for node in BRANCH_NODES:
        if node["id"] in visible_nodes:
            max_strength = node_max_incoming.get(node["id"], 0.5)
            opacity = float(np.interp(max_strength, [0, 1], [0.6, 1.0]))
            node_elements.append({
                "data": {
                    "id": node["id"],
                    "label": node["label"],
                    "group": node["group"],
                    "opacity": opacity,
                }
            })
    
    # Normalize edge strengths based on actual range
    if edges_with_strength:
        strengths = [e["strength"] for e in edges_with_strength]
        min_strength = min(strengths)
        max_strength = max(strengths)
        strength_range = max_strength - min_strength if max_strength > min_strength else 1.0
    else:
        min_strength = 0.0
        max_strength = 1.0
        strength_range = 1.0
    
    # Build edge elements
    edge_elements = []
    for idx, edge in enumerate(edges_with_strength):
        # Normalize strength to [0, 1] based on actual range
        normalized_strength = (edge["strength"] - min_strength) / strength_range if strength_range > 0 else 0.5
        
        # Map normalized strength to width [2px, 10px] and opacity [0.5, 1.0]
        width = float(np.interp(normalized_strength, [0, 1], [2, 10]))
        opacity = float(np.interp(normalized_strength, [0, 1], [0.5, 1.0]))
        
        # Get group from source node
        group = next((n["group"] for n in BRANCH_NODES if n["id"] == edge["source"]), "neutral")
        
        edge_elements.append({
            "data": {
                "id": f"{edge['source']}-{edge['target']}",
                "source": edge["source"],
                "target": edge["target"],
                "strength": float(edge["strength"]),
                "width": width,
                "opacity": opacity,
                "group": group,
            }
        })
    
    elements = node_elements + edge_elements
    
    # Build stylesheet
    stylesheet = [
        # Base node style
        {
            "selector": "node",
            "style": {
                "label": "data(label)",
                "font-size": "13px",
                "font-family": "'Inter', 'Segoe UI', system-ui, sans-serif",
                "color": pal["ink"],
                "text-halign": "center",
                "text-valign": "center",
                "background-color": pal["neutral"],
                "background-opacity": "data(opacity)",
                "width": "60px",
                "height": "32px",
                "shape": "roundrectangle",
                "border-width": 2,
                "border-color": pal["ink"],
                "border-opacity": 0.3,
            }
        },
        # Brown nodes
        {
            "selector": "node[group='brown']",
            "style": {
                "background-color": pal["low"],
                "color": pal["ink"],
                "border-color": pal["low"],
            }
        },
        # Sand nodes
        {
            "selector": "node[group='sand']",
            "style": {
                "background-color": pal["power"],
                "color": pal["ink"],
                "border-color": pal["power"],
            }
        },
        # Neutral nodes
        {
            "selector": "node[group='neutral']",
            "style": {
                "background-color": pal["ink"],
                "color": pal["bg"],
                "font-weight": "600",
                "border-color": pal["ink"],
            }
        },
        # Base edge style
        {
            "selector": "edge",
            "style": {
                "width": "data(width)",
                "line-color": pal["neutral"],
                "opacity": "data(opacity)",
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "target-arrow-color": pal["neutral"],
                "arrow-scale": 1.0,
            }
        },
        # Brown edges
        {
            "selector": "edge[group='brown']",
            "style": {
                "line-color": pal["low"],
                "target-arrow-color": pal["low"],
            }
        },
        # Sand edges
        {
            "selector": "edge[group='sand']",
            "style": {
                "line-color": pal["power"],
                "target-arrow-color": pal["power"],
            }
        },
        # Root/neutral edges
        {
            "selector": "edge[group='neutral']",
            "style": {
                "line-color": pal["ink"],
                "target-arrow-color": pal["ink"],
            }
        },
    ]
    
    return elements, stylesheet


# ---------- Scene 06: Frog Markov chain ----------

# Node positions for the triangle (equilateral, centered at origin)
FROG_NODES = {
    "A": (0, 1),
    "B": (-0.866, -0.5),
    "C": (0.866, -0.5),
}

# Default transition matrix (non-uniform)
DEFAULT_TRANSITION_MATRIX = [
    [0.1, 0.6, 0.3],  # From A: mostly to B
    [0.4, 0.2, 0.4],  # From B: equally to A and C
    [0.5, 0.3, 0.2],  # From C: mostly to A
]

# State colors
STATE_COLORS = {
    "A": "#2563eb",  # blue
    "B": "#f97316",  # orange
    "C": "#d946ef",  # magenta
}


def normalize_matrix_row(row: Sequence[float]) -> np.ndarray:
    """Normalize a row of probabilities to sum to 1."""
    arr = np.array(row, dtype=float)
    arr[arr < 0] = 0.0
    total = arr.sum()
    if total <= 0:
        return np.full(arr.shape, 1.0 / arr.size)
    return arr / total


def validate_transition_matrix(matrix: List[List[float]]) -> bool:
    """Check if transition matrix is valid (all rows sum to ~1, non-negative)."""
    if not matrix or len(matrix) != 3:
        return False
    for row in matrix:
        if len(row) != 3:
            return False
        if any(x < 0 for x in row):
            return False
        row_sum = sum(row)
        if abs(row_sum - 1.0) > 0.01:  # Allow small tolerance
            return False
    return True


def create_frog_node_graph(current_state: int, positions: dict, theme: str | None = None) -> go.Figure:
    """Create a triangle node graph with the frog's current position."""
    pal = get_palette(theme)
    fig = go.Figure()
    
    # Draw edges (triangle outline)
    states = ["A", "B", "C"]
    for i in range(3):
        j = (i + 1) % 3
        x0, y0 = positions[states[i]]
        x1, y1 = positions[states[j]]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=pal["axis"], width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    
    # Draw nodes with state colors
    for idx, state in enumerate(states):
        x, y = positions[state]
        is_current = (idx == current_state)
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker=dict(
                    size=40 if is_current else 30,
                    color=STATE_COLORS[state],
                    opacity=1.0 if is_current else 0.6,
                    line=dict(color="#ffffff", width=3 if is_current else 2),
                ),
                text=state,
                textposition="middle center",
                textfont=dict(
                    size=18 if is_current else 14, 
                    color="#ffffff", 
                    family="Inter",
                    weight="bold" if is_current else "normal"
                ),
                hovertemplate=f"State {state}<extra></extra>",
                showlegend=False,
            )
        )
    
    fig.update_layout(
        template=None,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showline=False,
            visible=False,
            range=[-1.2, 1.2],
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showline=False,
            visible=False,
            range=[-0.8, 1.3],
            scaleanchor="x",
            scaleratio=1,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=pal["ink"]),
        hoverlabel=dict(bgcolor=pal["bg"], font=dict(color=pal["ink"]), bordercolor=pal["axis"]),
    )
    
    # Add transition for animation
    fig.update_traces(
        selector=dict(mode="markers"),
    )
    
    return fig


def create_frog_pdf_chart(counts: Sequence[int], theme: str | None = None) -> go.Figure:
    """Create a bar chart showing normalized visit frequencies."""
    pal = get_palette(theme)
    states = ["A", "B", "C"]
    counts_arr = np.array(counts, dtype=float)
    total = counts_arr.sum()
    
    if total > 0:
        probabilities = counts_arr / total
    else:
        probabilities = np.zeros(3)
    
    # Use state colors for bars
    colors = [STATE_COLORS[state] for state in states]
    
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=states,
            y=probabilities,
            marker=dict(
                color=colors,
                line=dict(color=pal["bg"], width=2),
            ),
            hovertemplate="%{x}<br>frequency=%{y:.3f}<extra></extra>",
        )
    )
    
    fig.update_layout(
        template=None,
        margin=dict(l=20, r=20, t=10, b=40),
        showlegend=False,
        xaxis=dict(title="State", showgrid=False, linecolor=pal["axis"], tickcolor=pal["ink"], tickfont=dict(color=pal["ink"])),
        yaxis=dict(
            title="",
            showgrid=False,
            linecolor=pal["axis"],
            tickcolor=pal["ink"],
            tickfont=dict(color=pal["ink"]),
            showticklabels=False,
            visible=False,
            range=[0, 0.4],
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=pal["ink"]),
        hoverlabel=dict(bgcolor=pal["bg"], font=dict(color=pal["ink"]), bordercolor=pal["axis"]),
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
                        html.H1("Reasoning with Sampling"),
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
                                    max=10,
                                    step=0.1,
                                    value=4.0,
                                    marks={i: str(i) for i in range(1, 11)},
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
                    "RL post-training effectively sharpens the sequence sampling distribution",
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
                        html.Div(
                            className="playback-controls",
                            children=[
                                html.Button(
                                    "Pause",
                                    id="scene3-toggle-btn",
                                    n_clicks=0,
                                    className="playback-button toggle",
                                ),
                                html.Button(
                                    "Restart",
                                    id="scene3-restart-btn",
                                    n_clicks=0,
                                    className="playback-button restart",
                                ),
                            ],
                        ),
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
                        html.Div(
                            className="playback-controls",
                            children=[
                                html.Button(
                                    "Pause",
                                    id="scene4-toggle-btn",
                                    n_clicks=0,
                                    className="playback-button toggle",
                                ),
                                html.Button(
                                    "Restart",
                                    id="scene4-restart-btn",
                                    n_clicks=0,
                                    className="playback-button restart",
                                ),
                            ],
                        ),
                    ],
                ),
                card(
                    "scene-branching",
                    "Scene 05",
                    "Many weak paths vs few strong paths",
                    "",
                    [
                        cyto.Cytoscape(
                            id="branch-graph",
                            elements=[],
                            stylesheet=[],
                            layout={
                                "name": "breadthfirst",
                                "roots": "#root",
                                "directed": True,
                                "spacingFactor": 1.5,
                                "animate": False,
                            },
                            style={"width": "100%", "height": "500px"},
                            userZoomingEnabled=False,
                            userPanningEnabled=True,
                            boxSelectionEnabled=False,
                        ),
                    ],
                ),
                card(
                    "scene-frog",
                    "Scene 06",
                    "MCMC frog",
                    "",
                    [
                        html.Div(
                            className="frog-controls-grid",
                            children=[
                                html.Div(
                                    className="frog-column frog-controls-column",
                                    children=[
                                        html.Div(
                                            className="matrix-display-container",
                                            children=[
                                                html.P("Transition Matrix:", className="matrix-label"),
                                                html.Div(
                                                    className="matrix-display",
                                                    children=[
                                                        html.Div(className="matrix-display-row", children=[
                                                            html.Span("A→", className="matrix-display-label"),
                                                            html.Span("0.1", className="matrix-display-cell", style={"color": STATE_COLORS["A"]}),
                                                            html.Span("0.6", className="matrix-display-cell", style={"color": STATE_COLORS["B"]}),
                                                            html.Span("0.3", className="matrix-display-cell", style={"color": STATE_COLORS["C"]}),
                                                        ]),
                                                        html.Div(className="matrix-display-row", children=[
                                                            html.Span("B→", className="matrix-display-label"),
                                                            html.Span("0.4", className="matrix-display-cell", style={"color": STATE_COLORS["A"]}),
                                                            html.Span("0.2", className="matrix-display-cell", style={"color": STATE_COLORS["B"]}),
                                                            html.Span("0.4", className="matrix-display-cell", style={"color": STATE_COLORS["C"]}),
                                                        ]),
                                                        html.Div(className="matrix-display-row", children=[
                                                            html.Span("C→", className="matrix-display-label"),
                                                            html.Span("0.5", className="matrix-display-cell", style={"color": STATE_COLORS["A"]}),
                                                            html.Span("0.3", className="matrix-display-cell", style={"color": STATE_COLORS["B"]}),
                                                            html.Span("0.2", className="matrix-display-cell", style={"color": STATE_COLORS["C"]}),
                                                        ]),
                                                        html.Div(className="matrix-display-row matrix-display-header", children=[
                                                            html.Span("", className="matrix-display-label"),
                                                            html.Span("A", className="matrix-display-cell", style={"color": STATE_COLORS["A"]}),
                                                            html.Span("B", className="matrix-display-cell", style={"color": STATE_COLORS["B"]}),
                                                            html.Span("C", className="matrix-display-cell", style={"color": STATE_COLORS["C"]}),
                                                        ]),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="frog-speed-control",
                                            children=[
                                                html.P("Speed:", className="control-label"),
                                                dcc.Slider(
                                                    id="frog-speed-slider",
                                                    min=1,
                                                    max=15,
                                                    step=1,
                                                    value=5,
                                                    marks={1: "Slow", 5: "Medium", 10: "Fast", 15: "Rapido"},
                                                    className="frog-slider",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="frog-controls",
                                            children=[
                                                html.Button("Step", id="frog-step-btn", n_clicks=0, className="playback-button frog-btn"),
                                                html.Button("Run", id="frog-run-btn", n_clicks=0, className="playback-button toggle frog-btn"),
                                                html.Button("Reset", id="frog-reset-btn", n_clicks=0, className="playback-button restart frog-btn"),
                                            ],
                                        ),
                                        html.Div(
                                            id="frog-jumps-display",
                                            className="frog-jumps",
                                            children="Jumps: 0",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="frog-column frog-triangle-column",
                                    children=[
                                        dcc.Graph(id="frog-nodes-graph", config={"displayModeBar": False}),
                                    ],
                                ),
                                html.Div(
                                    className="frog-column frog-pdf-column",
                                    children=[
                                        dcc.Graph(id="frog-pdf-graph", config={"displayModeBar": False}),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                card(
                    "scene-hist-mh",
                    "Scene 07",
                    "Likelihoods revisited",
                    None,
                    [dcc.Graph(id="mh-hist-graph", config={"displayModeBar": False})],
                ),
            ],
        ),
        dcc.Store(id="theme-store", data="light"),
        dcc.Interval(id="scene3-interval", interval=3200, n_intervals=0),
        dcc.Store(id="base-prefix-store", data=INITIAL_BASE_SCENE3),
        dcc.Store(id="low-prefix-store", data=INITIAL_FULL_SCENE3),
        dcc.Store(id="scene3-playback-store", data=True),
        dcc.Interval(id="frog-interval", interval=400, n_intervals=0, disabled=True),
        dcc.Store(id="frog-state-store", data={"current": 0, "counts": [0, 0, 0], "jumps": 0, "running": False}),
    ]
)


# ---------- Callbacks ----------

# Clientside callback to detect system theme
# This triggers on initial page load via the alpha-slider value
app.clientside_callback(
    """
    function(alpha_value) {
        const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        return isDark ? 'dark' : 'light';
    }
    """,
    Output("theme-store", "data"),
    Input("alpha-slider", "value")
)

@app.callback(
    Output("distribution-graph", "figure"), 
    Output("alpha-label", "children"), 
    Input("alpha-slider", "value"),
    Input("theme-store", "data")
)
def update_distribution(alpha: float, theme: str):
    return distribution_figure(alpha, theme), f"{alpha:.1f}"


@app.callback(
    Output("rl-hist-graph", "figure"), 
    Input("alpha-slider", "value"),
    Input("theme-store", "data")
)
def init_rl_hist(_, theme: str):
    return likelihood_histogram(["base", "rl"], theme)


@app.callback(
    Output("mh-hist-graph", "figure"), 
    Input("alpha-slider", "value"),
    Input("theme-store", "data")
)
def init_mh_hist(_, theme: str):
    return likelihood_histogram(["base", "mh", "rl"], theme)


@app.callback(
    Output("scene3-playback-store", "data"),
    Input("scene3-toggle-btn", "n_clicks"),
    Input("scene4-toggle-btn", "n_clicks"),
    State("scene3-playback-store", "data"),
    prevent_initial_call=True,
)
def toggle_scene3_playback(_scene3_clicks, _scene4_clicks, current_state):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if triggered_id in {"scene3-toggle-btn", "scene4-toggle-btn"}:
        return not bool(current_state)
    return bool(current_state)


@app.callback(Output("scene3-interval", "disabled"), Input("scene3-playback-store", "data"))
def set_scene3_interval_disabled(is_playing):
    return not bool(is_playing)


@app.callback(
    Output("scene3-toggle-btn", "children"),
    Output("scene4-toggle-btn", "children"),
    Input("scene3-playback-store", "data"),
)
def update_toggle_labels(is_playing):
    label = "Pause" if is_playing else "Play"
    return label, label


@app.callback(
    Output("base-prefix-store", "data"),
    Output("low-prefix-store", "data"),
    Input("scene3-interval", "n_intervals"),
    Input("scene3-restart-btn", "n_clicks"),
    Input("scene4-restart-btn", "n_clicks"),
    State("base-prefix-store", "data"),
    State("low-prefix-store", "data"),
    State("alpha-slider", "value"),
)
def advance_prefixes(n_intervals, scene3_restart, scene4_restart, base_store, low_store, _alpha):
    ctx = dash.callback_context
    if not ctx.triggered:
        return base_store, low_store
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if triggered_id in {"scene3-restart-btn", "scene4-restart-btn"}:
        return INITIAL_BASE_SCENE3, INITIAL_FULL_SCENE3
    if triggered_id == "scene3-interval":
        if n_intervals is None or n_intervals == 0:
            return base_store, low_store
        return scene_step(base_store, "low"), scene_step(low_store, "full")
    return base_store, low_store


@app.callback(
    Output("base-next-graph", "figure"),
    Output("base-prefix-line", "children"),
    Input("base-prefix-store", "data"),
    Input("alpha-slider", "value"),
    Input("theme-store", "data"),
)
def render_base_next(store, _alpha, theme: str):
    pal = get_palette(theme)
    data = store or scene_step(None, "low")
    tokens = data.get("tokens", [])
    base_probs = np.array(data.get("base_probs", []), dtype=float)
    overlay = np.array(data.get("overlay_probs", []), dtype=float)
    figure = scene3_figure(tokens, base_probs, overlay, data.get("selected"), "Low-temp", pal["low"], theme)
    prefix = scene3_prefix_spans(data.get("suffix", []), variant="base")
    return figure, prefix


@app.callback(
    Output("low-temp-next-graph", "figure"),
    Output("low-temp-prefix-line", "children"),
    Input("low-prefix-store", "data"),
    Input("alpha-slider", "value"),
    Input("theme-store", "data"),
)
def render_low_temp_next(store, _alpha, theme: str):
    pal = get_palette(theme)
    data = store or scene_step(None, "full")
    tokens = data.get("tokens", [])
    base_probs = np.array(data.get("base_probs", []), dtype=float)
    overlay = np.array(data.get("overlay_probs", []), dtype=float)
    figure = scene3_figure(tokens, base_probs, overlay, data.get("selected"), "$p^{\\alpha}$", pal["power"], theme)
    prefix = scene3_prefix_spans(data.get("suffix", []), variant="power")
    return figure, prefix


@app.callback(
    Output("branch-graph", "elements"),
    Output("branch-graph", "stylesheet"),
    Input("alpha-slider", "value"),
    Input("theme-store", "data"),
)
def update_branch_graph(alpha, theme: str):
    elements, stylesheet = build_branch_elements(alpha, theme)
    return elements, stylesheet


# ---------- Frog Scene Callbacks ----------

@app.callback(
    Output("frog-interval", "interval"),
    Input("frog-speed-slider", "value"),
)
def update_frog_interval_speed(speed):
    """Update interval based on speed slider (1=slow, 15=very fast)."""
    # Map speed 1-15 to interval 1000ms-50ms
    if speed <= 10:
        interval = 1100 - (speed * 100)
    else:
        # For speeds 11-15, map to 100ms-50ms
        interval = 1100 - (speed * 100)
    return interval


@app.callback(
    Output("frog-state-store", "data"),
    Input("frog-reset-btn", "n_clicks"),
    Input("frog-step-btn", "n_clicks"),
    Input("frog-interval", "n_intervals"),
    State("frog-state-store", "data"),
    prevent_initial_call=True,
)
def update_frog_state(reset_clicks, step_clicks, n_intervals, state_data):
    """Handle reset, step, and interval-triggered steps."""
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # Reset - always start from A (state 0)
    if triggered_id == "frog-reset-btn":
        return {
            "current": 0,
            "counts": [0, 0, 0],
            "jumps": 0,
            "running": False,
        }
    
    # Step or interval tick
    if triggered_id in ["frog-step-btn", "frog-interval"]:
        # Get current state
        current = state_data.get("current", 0)
        counts = list(state_data.get("counts", [0, 0, 0]))
        jumps = state_data.get("jumps", 0)
        running = state_data.get("running", False)
        
        # Get transition probabilities for current state (fixed matrix)
        transition_probs = DEFAULT_TRANSITION_MATRIX[current]
        
        # Sample next state
        rng = np.random.default_rng()
        next_state = rng.choice(3, p=transition_probs)
        
        # Update counts and jumps
        counts[next_state] += 1
        jumps += 1
        
        return {
            "current": int(next_state),
            "counts": counts,
            "jumps": jumps,
            "running": running,
        }
    
    raise PreventUpdate


@app.callback(
    Output("frog-state-store", "data", allow_duplicate=True),
    Input("frog-run-btn", "n_clicks"),
    State("frog-state-store", "data"),
    prevent_initial_call=True,
)
def toggle_frog_running(n_clicks, state_data):
    """Toggle the running state."""
    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate
    
    state_data = state_data or {"current": 0, "counts": [0, 0, 0], "jumps": 0, "running": False}
    state_data["running"] = not state_data.get("running", False)
    return state_data


@app.callback(
    Output("frog-interval", "disabled"),
    Input("frog-state-store", "data"),
)
def control_frog_interval(state_data):
    """Enable/disable interval based on running state."""
    running = state_data.get("running", False) if state_data else False
    return not running


@app.callback(
    Output("frog-run-btn", "children"),
    Input("frog-state-store", "data"),
)
def update_frog_run_button_label(state_data):
    """Update the Run/Pause button label."""
    running = state_data.get("running", False) if state_data else False
    return "Pause" if running else "Run"


@app.callback(
    Output("frog-nodes-graph", "figure"),
    Input("frog-state-store", "data"),
    Input("theme-store", "data"),
)
def update_frog_nodes_graph(state_data, theme: str):
    """Update the node graph with frog position."""
    current_state = state_data.get("current", 0) if state_data else 0
    return create_frog_node_graph(current_state, FROG_NODES, theme)


@app.callback(
    Output("frog-pdf-graph", "figure"),
    Input("frog-state-store", "data"),
    Input("theme-store", "data"),
)
def update_frog_pdf_graph(state_data, theme: str):
    """Update the PDF bar chart."""
    counts = state_data.get("counts", [0, 0, 0]) if state_data else [0, 0, 0]
    return create_frog_pdf_chart(counts, theme)


@app.callback(
    Output("frog-jumps-display", "children"),
    Input("frog-state-store", "data"),
)
def update_frog_jumps_display(state_data):
    """Update the jumps counter display."""
    jumps = state_data.get("jumps", 0) if state_data else 0
    return f"Jumps: {jumps}"


if __name__ == "__main__":
    app.run(debug=True)
