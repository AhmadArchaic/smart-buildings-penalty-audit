import csv, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = [r for r in csv.DictReader(open("results_round2.csv")) if r["length"] == "F"]
clim = ["atlanta", "denver", "fairbanks", "phoenix", "singapore"]
pols = [("pid_water", "pid_water"), ("CtrlX", "CtrlX")]
fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.3))
for ax, c in zip(axes, clim):
    for i, (p, lab) in enumerate(pols):
        rs = [r for r in R if r["policy"] == p and c in r["scenario"]]
        parts = [st.mean(0.6 * float(r["comfort_regret"]) for r in rs), st.mean(0.2 * float(r["energy"]) for r in rs), st.mean(0.2 * float(r["carbon"]) for r in rs)]
        b = 0
        for v, col, l in zip(parts, ["#c44e52", "#4c72b0", "#8c8c8c"], ["comfort", "energy cost", "carbon"]):
            ax.bar(i, v, bottom=b, color=col, width=0.6, label=l if (c == "atlanta" and i == 0) else None); b += v
    ax.set_xticks([0, 1]); ax.set_xticklabels([l for _, l in pols], fontsize=6.5)
    ax.set_title(c.capitalize(), fontsize=8); ax.tick_params(labelsize=6.5)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("mean score (lower is better)", fontsize=7)
fig.legend(loc="upper center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.04))
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig("fig_anatomy.pdf"); print("ok")
