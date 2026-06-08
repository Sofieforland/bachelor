import matplotlib.pyplot as plt

# ---- DATA ----
data = {
    "LLaMA": {
        "No Rep": {
            "Cautious": {"influence": 0.806, "bal_acc": 0.500},
            "Conservative": {"influence": 0.257, "bal_acc": 0.500},
            "Neutral": {"influence": 0.566, "bal_acc": 0.702},
            "Overconfident": {"influence": 0.213, "bal_acc": 0.847},
        },
        "Rep 0": {
            "Cautious": {"influence": 0.735, "bal_acc": 0.500},
            "Conservative": {"influence": 0.316, "bal_acc": 0.500},
            "Neutral": {"influence": 0.545, "bal_acc": 0.702},
            "Overconfident": {"influence": 0.205, "bal_acc": 0.847},
        },
        "Rep 1": {
            "Cautious": {"influence": 0.858, "bal_acc": 0.500},
            "Conservative": {"influence": 0.208, "bal_acc": 0.500},
            "Neutral": {"influence": 0.402, "bal_acc": 0.702},
            "Overconfident": {"influence": 0.298, "bal_acc": 0.847},
        },
    },

    "Qwen": {
        "No Rep": {
            "Cautious": {"influence": 0.977, "bal_acc": 0.526},
            "Conservative": {"influence": 0.661, "bal_acc": 0.510},
            "Neutral": {"influence": 0.136, "bal_acc": 0.707},
            "Overconfident": {"influence": 0.369, "bal_acc": 0.730},
        },
        "Rep 0": {
            "Cautious": {"influence": 0.974, "bal_acc": 0.526},
            "Conservative": {"influence": 0.652, "bal_acc": 0.510},
            "Neutral": {"influence": 0.133, "bal_acc": 0.707},
            "Overconfident": {"influence": 0.364, "bal_acc": 0.730},
        },
        "Rep 1": {
            "Cautious": {"influence": 0.983, "bal_acc": 0.526},
            "Conservative": {"influence": 0.471, "bal_acc": 0.510},
            "Neutral": {"influence": 0.071, "bal_acc": 0.707},
            "Overconfident": {"influence": 0.483, "bal_acc": 0.730},
        },
    },

    "MedGemma": {
        "No Rep": {
            "Cautious": {"influence": 0.789, "bal_acc": 0.565},
            "Conservative": {"influence": 0.531, "bal_acc": 0.532},
            "Neutral": {"influence": 0.783, "bal_acc": 0.612},
            "Overconfident": {"influence": 0.760, "bal_acc": 0.646},
        },
        "Rep 0": {
            "Cautious": {"influence": 0.795, "bal_acc": 0.565},
            "Conservative": {"influence": 0.526, "bal_acc": 0.532},
            "Neutral": {"influence": 0.782, "bal_acc": 0.612},
            "Overconfident": {"influence": 0.789, "bal_acc": 0.646},
        },
        "Rep 1": {
            "Cautious": {"influence": 0.784, "bal_acc": 0.565},
            "Conservative": {"influence": 0.529, "bal_acc": 0.532},
            "Neutral": {"influence": 0.772, "bal_acc": 0.612},
            "Overconfident": {"influence": 0.742, "bal_acc": 0.646},
        },
    }
}

# ---- STYLING ----
colors = {"No Rep": "blue", "Rep 0": "green", "Rep 1": "red"}
markers = {"Cautious": "o", "Conservative": "s", "Neutral": "^", "Overconfident": "D"}

# ---- PLOT ----
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

for ax, (model_name, model_data) in zip(axes, data.items()):
    for setting, gps in model_data.items():
        for gp, vals in gps.items():
            ax.scatter(
                vals["influence"],
                vals["bal_acc"],
                color=colors[setting],
                marker=markers[gp],
                s=70
            )

    ax.set_title(model_name)
    ax.grid(True)

# Axis labels (kun én gang for clean look)
axes[0].set_ylabel("Balanced Accuracy")
for ax in axes:
    ax.set_xlabel("Influence")

# ---- LEGENDS ----
# Farger (settings)
for setting, color in colors.items():
    axes[0].scatter([], [], color=color, label=setting)

# Markører (GPs)
for gp, marker in markers.items():
    axes[0].scatter([], [], color="black", marker=marker, label=gp)

fig.legend(loc="upper center", ncol=7)
plt.tight_layout(rect=[0, 0, 1, 0.92])

plt.savefig("subplot_models.pdf", bbox_inches="tight")
plt.show()