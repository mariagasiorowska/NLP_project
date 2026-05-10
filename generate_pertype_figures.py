import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"font.family":"serif","font.size":9,
    "axes.titlesize":9,"axes.labelsize":9,"xtick.labelsize":8,
    "ytick.labelsize":8,"legend.fontsize":8})
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

splits = ["Standard","Ent-disj.","Freq-adv.","Ctx-shift","Cross-dom."]
models = ["DistilBERT","BiLSTM-CRF","Llama ZS","Llama FT"]
colors = ["#4C72B0","#DD8452","#55A868","#C44E52"]
markers = ["s","o","^","D"]
ls = ["-","-","--","--"]

# PER / LOC / ORG F1 — rows=models, cols=splits
per = np.array([
    [0.8670, 0.7404, 0.7452, 0.7226, 0.8540],  # DistilBERT
    [0.5546, 0.0557, 0.0850, 0.0811, 0.1987],  # BiLSTM
    [0.0080, 0.0318, 0.0386, 0.0060, 0.0197],  # Llama ZS
    [0.3794, 0.1051, 0.1948, 0.0100, 0.2281],  # Llama FT
])
loc = np.array([
    [0.8204, 0.4669, 0.4905, 0.7046, 0.8657],  # DistilBERT
    [0.5849, 0.0629, 0.0172, 0.1330, 0.3953],  # BiLSTM
    [0.0097, 0.0175, 0.0235, 0.0053, 0.0052],  # Llama ZS
    [0.2944, 0.0133, 0.0504, 0.0122, 0.1603],  # Llama FT
])
org = np.array([
    [0.5788, 0.2097, 0.1164, 0.2916, 0.4512],  # DistilBERT
    [0.2970, 0.0376, 0.0099, 0.0121, 0.0579],  # BiLSTM
    [0.0012, 0.0088, 0.0085, 0.0021, 0.0033],  # Llama ZS
    [0.0465, 0.0117, 0.0122, 0.0017, 0.0479],  # Llama FT
])

x = np.arange(len(splits))

for data, entity, fname in [(per,"PER","per_f1.pdf"),
                             (loc,"LOC","loc_f1.pdf"),
                             (org,"ORG","org_f1.pdf")]:
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    for i,(m,c,mk,l) in enumerate(zip(models,colors,markers,ls)):
        ax.plot(x, data[i], color=c, marker=mk, linestyle=l,
                linewidth=1.5, markersize=5, label=m)
    ax.set_xticks(x); ax.set_xticklabels(splits)
    ax.set_ylabel(f"Span-level F1 ({entity})")
    ax.set_ylim(-0.02, 1.0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.title(f"{entity} span F1 across evaluation conditions", pad=6)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, fname)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

# Combined 3-panel figure
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), sharey=False)
for ax, data, entity in zip(axes,
                             [per, loc, org],
                             ["PER", "LOC", "ORG"]):
    for i,(m,c,mk,l) in enumerate(zip(models,colors,markers,ls)):
        ax.plot(x, data[i], color=c, marker=mk, linestyle=l,
                linewidth=1.4, markersize=4, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(["Std","ED","FA","CS","CD"], fontsize=7)
    ax.set_title(entity, fontsize=9)
    ax.set_ylim(-0.02, 1.0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Span-level F1")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4,
           fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.08))
plt.suptitle("Per-entity-type F1 across evaluation conditions",
             fontsize=9, y=1.01)
plt.tight_layout()
combined_path = os.path.join(FIGURES_DIR, "pertype_combined.pdf")
plt.savefig(combined_path, bbox_inches="tight")
plt.close()
print(f"Saved {combined_path}")