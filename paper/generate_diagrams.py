"""Generate the two architecture diagrams referenced by paper/main.tex.

Outputs (under ``paper/Figures/`` by default):
  * ``overview_atlas.png``      — splash topology figure (5 fusion centers,
    dashed federation lines to a central FL host, geographic backdrop).
  * ``atlas_architecture.png``  — three-layer × three-column matrix:
        Layer 1 (Privacy-Preserving Partitioning)        — yellow
        Layer 2 (Multi-Task FUSION-MLP)                   — green
        Layer 3 (Federated Coordination)                   — blue
      columns: per-fusion-center data | federation clients | FL host

Visual style is adapted from the HERMES overview/architecture pages of
``HERMES Overview (2).drawio``: same swimlane color palette, same
column-header band, same dashed cross-layer arrows.

Usage:
    python paper/generate_diagrams.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
import numpy as np


# ---------------------------------------------------------------------------
# Style palette — matches the HERMES drawio layer palette
# ---------------------------------------------------------------------------
LAYER1_FILL  = "#fffcf0"   # cream
LAYER1_EDGE  = "#d6b656"
LAYER1_TEXT  = "#9a6c00"

LAYER2_FILL  = "#f5fbf5"   # mint
LAYER2_EDGE  = "#82b366"
LAYER2_TEXT  = "#3d6b48"

LAYER3_FILL  = "#f0f5ff"   # ice blue
LAYER3_EDGE  = "#6c8ebf"
LAYER3_TEXT  = "#2d4f80"

HEADER_FILL  = "#2d3748"   # charcoal column headers
HEADER_TEXT  = "white"

CLIENT_COLORS = ["#B5739D", "#B9E0A5", "#7EA6E0", "#D79B00", "#B85450"]

BLOCK_FILL   = "#ffffff"
BLOCK_EDGE   = "#444444"


# ---------------------------------------------------------------------------
# Overview figure  (analog of HERMES drawio page 4)
# ---------------------------------------------------------------------------
def make_overview(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    # Compact title flush to the top
    ax.text(0.50, 0.965,
            "ATLAS: Cross-jurisdictional federated threat-intelligence training",
            ha="center", va="center", fontsize=12, fontweight="bold")

    # FL host node
    central_host = (0.50, 0.82)
    host_w, host_h = 0.22, 0.11
    host_box = FancyBboxPatch(
        (central_host[0] - host_w/2, central_host[1] - host_h/2),
        host_w, host_h,
        boxstyle="round,pad=0.015", facecolor=HEADER_FILL,
        edgecolor=HEADER_FILL, linewidth=2, zorder=4,
    )
    ax.add_patch(host_box)
    ax.text(*central_host, "FL Host Process\n(FedAvg / FedProx)\nHost server",
            ha="center", va="center", fontsize=9.5, color=HEADER_TEXT,
            fontweight="bold", zorder=5)

    # Subtitle, placed between the host and the client row, with a soft
    # translucent backdrop so it reads cleanly through the dashed lines
    ax.text(0.50, 0.625,
            "Only model updates flow over the dashed links — raw records never leave a fusion federated client.",
            ha="center", va="center", fontsize=9, color="#444", style="italic",
            zorder=7,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2.5))

    # Geographic backdrop band
    backdrop = mpatches.Polygon(
        [(0.04, 0.06), (0.96, 0.06), (0.98, 0.56), (0.02, 0.56)],
        closed=True, facecolor="#eef4fa", edgecolor="#aec3da",
        linewidth=1.5, zorder=1,
    )
    ax.add_patch(backdrop)

    # Backdrop label — layered above lines with translucent box
    ax.text(0.50, 0.520, "U.S. fusion-center geography (FIPS-derived partition)",
            ha="center", va="center", fontsize=9, style="italic",
            color="#5a738f", zorder=7,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))

    # Five fusion-federated-client nodes on a single horizontal row
    client_y = 0.27
    client_w, client_h = 0.175, 0.18
    centers = [
        (0.115, client_y, "Fusion Federated\nClient 0  (NW)"),
        (0.308, client_y, "Fusion Federated\nClient 1  (West)"),
        (0.500, client_y, "Fusion Federated\nClient 2  (Central)"),
        (0.692, client_y, "Fusion Federated\nClient 3  (South)"),
        (0.885, client_y, "Fusion Federated\nClient 4  (NE)"),
    ]

    # Dashed federation connections — host bottom → client top, fanned
    host_bottom = (central_host[0], central_host[1] - host_h/2)
    for (x, y, _), color in zip(centers, CLIENT_COLORS):
        dx = x - central_host[0]
        rad = 0.18 * np.sign(dx) * min(abs(dx) / 0.40, 1.0)
        ax.add_patch(FancyArrowPatch(
            host_bottom, (x, y + client_h/2),
            arrowstyle="-", linestyle="--", linewidth=2.2,
            color=color, connectionstyle=f"arc3,rad={rad}",
            zorder=3,
        ))

    # Fusion-federated-client rounded boxes
    for (x, y, label), color in zip(centers, CLIENT_COLORS):
        client_box = FancyBboxPatch(
            (x - client_w/2, y - client_h/2), client_w, client_h,
            boxstyle="round,pad=0.012", facecolor="white",
            edgecolor=color, linewidth=2.5, zorder=5,
        )
        ax.add_patch(client_box)
        ax.text(x, y + 0.030, label, ha="center", va="center",
                fontsize=8, fontweight="bold", color=color, zorder=6)
        ax.text(x, y - 0.035, "[local data stays here]",
                ha="center", va="center",
                fontsize=6.5, color="#555", style="italic", zorder=6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Architecture figure  (analog of HERMES drawio page 5)
# ---------------------------------------------------------------------------
def add_block(ax, x, y, w, h, title, lines, edge=BLOCK_EDGE, fill=BLOCK_FILL):
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.005",
        facecolor=fill, edgecolor=edge, linewidth=1.6, zorder=4,
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.025, title,
            ha="center", va="top", fontsize=9, fontweight="bold", zorder=5)
    body = "\n".join(lines)
    ax.text(x + w/2, y + h/2 - 0.01, body,
            ha="center", va="center", fontsize=7.5, zorder=5, color="#333")


def make_architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 8.5), dpi=300)

    # ----- column headers ------------------------------------------------
    col_x = [0.04, 0.36, 0.68]
    col_w = 0.28
    header_y, header_h = 0.91, 0.05
    headers = ["Per-Fusion-Center Data", "Federation Clients", "FL Host Process"]
    for x, lbl in zip(col_x, headers):
        ax.add_patch(Rectangle((x, header_y), col_w, header_h,
                               facecolor=HEADER_FILL, edgecolor=HEADER_FILL, zorder=3))
        ax.text(x + col_w/2, header_y + header_h/2, lbl,
                ha="center", va="center", fontsize=12, color=HEADER_TEXT,
                fontweight="bold", zorder=4)

    # ----- three layered swimlanes --------------------------------------
    layers = [
        ("Layer 1\nPrivacy-Preserving\nPartitioning",
         LAYER1_FILL, LAYER1_EDGE, LAYER1_TEXT, 0.66, 0.20),
        ("Layer 2\nMulti-Task\nFUSION-MLP",
         LAYER2_FILL, LAYER2_EDGE, LAYER2_TEXT, 0.40, 0.24),
        ("Layer 3\nFederated\nCoordination",
         LAYER3_FILL, LAYER3_EDGE, LAYER3_TEXT, 0.10, 0.28),
    ]
    lane_x = col_x[0]
    lane_w = (col_x[2] + col_w) - lane_x

    for label, fill, edge, txt, y, h in layers:
        ax.add_patch(Rectangle((lane_x - 0.005, y), lane_w + 0.01, h,
                               facecolor=fill, edgecolor=edge, linewidth=2, zorder=1))
        ax.text(lane_x - 0.025, y + h/2, label,
                ha="right", va="center", fontsize=10, color=txt,
                fontweight="bold", fontstyle="italic", zorder=2)

    # ----- Layer 1 blocks (yellow lane y=0.66, h=0.20) ------------------
    add_block(ax, col_x[0] + 0.01, 0.70, col_w - 0.02, 0.13,
              "UCI Communities & Crime",
              ["2,215 rows × 147 features",
               "FIPS state code → region bucket",
               "Drop 12 sensitive demographic cols",
               "Frozen 15% global test set"])
    add_block(ax, col_x[1] + 0.01, 0.70, col_w - 0.02, 0.13,
              "Per-client partition + scaler",
              ["StandardScaler fit locally",
               "(or --global_scaler shared)",
               "Geographic / IID / Dirichlet",
               "Deterministic from seed=42"])
    add_block(ax, col_x[2] + 0.01, 0.70, col_w - 0.02, 0.13,
              "Partition-stats audit log",
              ["partition_stats.json",
               "Per-client class distribution",
               "Dropped sensitive columns",
               "Used by figure scripts"])

    # ----- Layer 2 blocks (green lane y=0.40, h=0.24) -------------------
    add_block(ax, col_x[0] + 0.01, 0.46, col_w - 0.02, 0.16,
              "Local training step",
              ["Standardized x ∈ ℝ^d",
               "Mini-batch SGD (lr=1e-3)",
               "5 local epochs per round",
               "FedProx adds proximal grad"])
    add_block(ax, col_x[1] + 0.01, 0.46, col_w - 0.02, 0.16,
              "FUSION-MLP architecture",
              ["Shared 3-layer MLP trunk",
               "Threat head: 3-class softmax",
               "Escalation head: sigmoid",
               "Joint loss: α·CE + β·BCE",
               "|θ| ≈ 108 KB / round / client"])
    add_block(ax, col_x[2] + 0.01, 0.46, col_w - 0.02, 0.16,
              "Server-side eval",
              ["Eval on frozen global test set",
               "macro-F1, accuracy",
               "Escalation MAE/AUROC/Spearman",
               "fairness_macro_f1_variance",
               "proximal_contribution Π"])

    # ----- Layer 3 blocks (blue lane y=0.10, h=0.28) --------------------
    add_block(ax, col_x[0] + 0.01, 0.18, col_w - 0.02, 0.18,
              "TrainingClient.py",
              ["FusionMLPClient (Flower NumPy)",
               "Loads partition + scaler",
               "Returns updated weights θ_j",
               "Logs per-client metrics",
               "Federated or Central mode"])
    add_block(ax, col_x[1] + 0.01, 0.18, col_w - 0.02, 0.18,
              "Simulation OR Distributed",
              ["fl.simulation.start_simulation",
               "  (Ray actors, 1 process)",
               "        OR",
               "fl.client.start_numpy_client",
               "  (gRPC, per-machine)",
               "Same partition contract"])
    add_block(ax, col_x[2] + 0.01, 0.18, col_w - 0.02, 0.18,
              "HFLHost.py",
              ["FusionFedAvg (extends FedAvg)",
               "Weighted aggregation by |D_j|",
               "FedProx broadcasts μ in fit cfg",
               "Plateau detector (10-round)",
               "server_evaluation.log per round"])

    # ----- cross-layer arrows -------------------------------------------
    def vert_arrow(x1, y1, x2, y2, color="#666", style="-|>"):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle=style,
            mutation_scale=18, color=color, linewidth=1.6, zorder=6,
        ))

    # Layer 1 -> Layer 2
    for x in col_x:
        vert_arrow(x + col_w/2, 0.70, x + col_w/2, 0.62, color=LAYER1_EDGE)
    # Layer 2 -> Layer 3
    for x in col_x:
        vert_arrow(x + col_w/2, 0.46, x + col_w/2, 0.36, color=LAYER2_EDGE)
    # Round trip (host -> clients in Layer 3)
    ax.add_patch(FancyArrowPatch(
        (col_x[2] + col_w/2 - 0.05, 0.215),
        (col_x[0] + col_w/2 + 0.05, 0.215),
        arrowstyle="<|-|>", mutation_scale=18,
        color=LAYER3_EDGE, linewidth=1.8, zorder=6,
        connectionstyle="arc3,rad=-0.18",
    ))
    ax.text(0.50, 0.13, "global model θᵗ broadcast  ↔  client updates {θⱼ}",
            ha="center", va="center", fontsize=8.5,
            color=LAYER3_TEXT, style="italic", zorder=7)

    # Title
    ax.text(0.50, 0.985,
            "ATLAS: Three-Layer Coordination Framework",
            ha="center", va="center", fontsize=13, fontweight="bold")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir",
                        default=str(Path(__file__).resolve().parent / "Figures"),
                        help="Output directory (default: paper/Figures).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    over = out_dir / "overview_atlas.png"
    arch = out_dir / "atlas_architecture.png"

    print(f"Generating diagrams into {out_dir}")
    make_overview(over)
    print(f"  [OK] {over.name}")
    make_architecture(arch)
    print(f"  [OK] {arch.name}")
    print("Done.")


if __name__ == "__main__":
    main()
