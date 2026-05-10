"""
generate_figures.py
-------------------
Generates all figures for the NLP project report.
Run from your project folder:
    python generate_figures.py

Outputs (saved to figures/ folder):
    figures/f1_scores.pdf       — main results grouped bar chart
    figures/degradation.pdf     — F1 drop from standard split
    figures/heatmap.pdf         — F1 heatmap across models × splits
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10

os.makedirs("figures", exist_ok=True)

# ── DATA ──────────────────────────────────────────────────────────────────────
models = ["BiLSTM-CRF", "DistilBERT", "Llama ZS", "Llama FT"]
splits = ["Standard", "Ent-disj.", "Freq-adv.", "Ctx-shift", "Cross-dom."]

# F1 scores — rows = models, cols = splits
f1 = np.array([
    [0.502, 0.049, 0.020, 0.088, 0.254],   # BiLSTM-CRF
    [0.782, 0.533, 0.551, 0.634, 0.769],   # DistilBERT
    [0.007, 0.020, 0.025, 0.005, 0.010],   # Llama ZS
    [0.281, 0.052, 0.117, 0.009, 0.164],   # Llama FT
])

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# ── FIGURE 1: Grouped bar chart of F1 scores ──────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 3.2))

x = np.arange(len(splits))
width = 0.18
offsets = [-1.5, -0.5, 0.5, 1.5]

for i, (model, color, offset) in enumerate(zip(models, colors, offsets)):
    bars = ax.bar(x + offset * width, f1[i], width,
                  label=model, color=color, alpha=0.85, edgecolor='white')

ax.set_ylabel("Span-level F1")
ax.set_xticks(x)
ax.set_xticklabels(splits, fontsize=9)
ax.set_ylim(0, 1.0)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title("F1 scores across models and evaluation conditions", fontsize=10)
plt.tight_layout()
plt.savefig("figures/f1_scores.pdf", bbox_inches='tight')
plt.close()
print("Saved figures/f1_scores.pdf")

# ── FIGURE 2: F1 degradation (drop from standard split) ──────────────────────
fig, ax = plt.subplots(figsize=(6.5, 3.2))

degradation = f1[:, 0:1] - f1[:, 1:]   # standard minus each shift split
shift_splits = splits[1:]               # exclude standard

x = np.arange(len(shift_splits))
offsets = [-1.5, -0.5, 0.5, 1.5]

for i, (model, color, offset) in enumerate(zip(models, colors, offsets)):
    ax.bar(x + offset * width, degradation[i], width,
           label=model, color=color, alpha=0.85, edgecolor='white')

ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel("F1 drop from standard split")
ax.set_xticks(x)
ax.set_xticklabels(shift_splits, fontsize=9)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title("F1 degradation from standard split baseline", fontsize=10)
plt.tight_layout()
plt.savefig("figures/degradation.pdf", bbox_inches='tight')
plt.close()
print("Saved figures/degradation.pdf")

# ── FIGURE 3: Heatmap of F1 scores ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.0, 2.8))

im = ax.imshow(f1, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.85)

ax.set_xticks(np.arange(len(splits)))
ax.set_xticklabels(splits, fontsize=9)
ax.set_yticks(np.arange(len(models)))
ax.set_yticklabels(models, fontsize=9)

# annotate cells
for i in range(len(models)):
    for j in range(len(splits)):
        val = f1[i, j]
        color = "white" if val > 0.5 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=8, color=color, fontweight="bold")

plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="F1")
plt.title("Span-level F1 heatmap", fontsize=10)
plt.tight_layout()
plt.savefig("figures/heatmap.pdf", bbox_inches='tight')
plt.close()
print("Saved figures/heatmap.pdf")

print("\nAll figures saved to figures/")
print("Upload figures/ folder to Overleaf alongside report.tex")
