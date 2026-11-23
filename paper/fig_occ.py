import csv, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
clim = ["atlanta", "denver", "fairbanks", "phoenix", "singapore"]
A = list(csv.DictReader(open("occ30.csv")))
R = list(csv.DictReader(open("occ_release_30.csv")))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1, 1.25]})
sig = [("temp_wh", "zone temperature", "#dd8452"), ("damper_wh", "damper command", "#4c72b0"), ("arx_w6_wh", "model residual", "#55a868")]
w = 0.26
for j, (k, lab, col) in enumerate(sig):
    vals = [st.mean(float(r[k]) for r in A if c in r["scenario"]) for c in clim]
    a1.bar([i + (j - 1) * w for i in range(5)], vals, w, color=col, label=lab)
a1.axhline(0.5, color="k", lw=0.8, ls="--"); a1.text(-0.45, 0.503, "chance", fontsize=6, ha="left", va="bottom")
a1.set_ylim(0.45, 0.75); a1.set_xticks(range(5)); a1.set_xticklabels([c.capitalize() for c in clim], fontsize=6.5)
a1.set_ylabel("AUC, occupied vs empty\n(same hour of day)", fontsize=7); a1.tick_params(labelsize=6.5)
a1.legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(0.12, 1)); a1.set_title("(a) Is a person visible in a zone?", fontsize=8)
rel = lambda r, k: 100 * (float(r[k]) - float(r["ctrlx"])) / float(r["ctrlx"])
for i, c in enumerate(clim):
    rs = [r for r in R if c in r["scenario"]]
    a2.scatter([rel(r, "oracle_release") for r in rs], [i - 0.12] * len(rs), s=12, color="#55a868", label="true occupancy" if i == 0 else None)
    a2.scatter([rel(r, "estimator_release_1K") for r in rs], [i + 0.12] * len(rs), s=12, color="#c44e52", marker="s", label="our estimator" if i == 0 else None)
a2.axvline(0, color="k", lw=0.8)
a2.set_xscale("symlog", linthresh=10); a2.set_xticks([-10, 0, 10, 100, 1000]); a2.set_xticklabels(["$-10$", "0", "+10", "+100", "+1000"])
a2.set_yticks(range(5)); a2.set_yticklabels([c.capitalize() for c in clim], fontsize=6.5); a2.invert_yaxis()
a2.set_xlabel("score change vs CtrlX (%), lower is better", fontsize=7, labelpad=1); a2.tick_params(labelsize=6.5)
a2.legend(fontsize=6, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.22)); a2.set_title("(b) Afternoon release driven by ...", fontsize=8)
for a in (a1, a2): a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("fig_occ.pdf"); fig.savefig("fig_occ.png", dpi=150); print("ok")
