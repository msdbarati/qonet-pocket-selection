import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Computer Modern ("Modern Math") look for all figures ---
plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
})

import os
OUT = os.path.dirname(os.path.abspath(__file__))   # read/write next to this script
DST = OUT
res = json.load(open(f"{OUT}/results_antideriv.json"))
dep = json.load(open(f"{OUT}/depth.json"))

short = {"identity N(0,.05)":"identity","small-Gauss N(0,.3)":"small-Gauss",
         "calc_a U[-.4pi,.4pi]":"calc-a","shifted N(pi/4,.3)":"shifted","BP U[-pi,pi]":"BP-uniform"}
col   = {"identity N(0,.05)":"#1f77b4","small-Gauss N(0,.3)":"#2ca02c",
         "calc_a U[-.4pi,.4pi]":"#ff7f0e","shifted N(pi/4,.3)":"#9467bd","BP U[-pi,pi]":"#d62728"}

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

# Panel A: first-moment depth contraction (stage-1 validity gate)
for f, vs in dep["var"].items():
    ax[0].semilogy(dep["depths"], vs, marker="o", ms=5, color=col[f],
                   label=f"{short[f]} (E[cos]={dep['ecos'][f]:.2f})")
ax[0].set_xlabel("re-uploading depth L"); ax[0].set_ylabel(r"Var$[\partial\langle Z_0\rangle/\partial\theta_1]$")
ax[0].set_title(f"(a) Stage-1 gate: first-moment contraction (n={dep['n']})")
ax[0].grid(True, which="both", alpha=0.3); ax[0].legend(fontsize=8)

# Panel B: pocket score predicts test error (stage-2 criterion)
# per-family label offsets (points) and alignment to avoid any overlap
LBL = {"identity N(0,.05)":   (10,  4, "left"),
       "small-Gauss N(0,.3)": (10,  4, "left"),
       "shifted N(pi/4,.3)":  (-9,  -3, "right"),   # left of the dot
       "calc_a U[-.4pi,.4pi]":(-12, 15, "right"),   # top, slightly left
       "BP U[-pi,pi]":        (10, 17, "left")}     # top
for f, r in res.items():
    mk = "o" if r["valid"] else "X"
    ax[1].errorbar(r["S"], r["test_mean"], yerr=r["test_std"], marker=mk, ms=11, capsize=4,
                   color=col[f], mfc=(col[f] if r["valid"] else "none"), mew=2,
                   label=f"{short[f]}" + ("" if r["valid"] else " (gated out)"))
    dx, dy, ha = LBL[f]
    lbl = short[f] + ("" if r["valid"] else " (gated out)")
    ax[1].annotate(lbl, (r["S"], r["test_mean"]), textcoords="offset points",
                   xytext=(dx, dy), ha=ha, fontsize=8)
# trend over valid families
valid = [(r["S"], r["test_mean"]) for r in res.values() if r["valid"]]
valid.sort()
vx, vy = zip(*valid)
ax[1].plot(vx, vy, "--", color="gray", lw=1, alpha=0.7, zorder=0)
ax[1].set_xscale("log"); ax[1].set_xlim(2e-5, 1.6e-1)
ax[1].set_xlabel(r"pocket score  $S=(D\cdot T)^{1/2}$"); ax[1].set_ylabel(r"test relative $L_2$ error")
ax[1].set_title("(b) Stage-2 criterion: $S$ predicts downstream error")
ax[1].grid(True, which="both", alpha=0.3); ax[1].legend(fontsize=8, loc="upper right")

plt.tight_layout()
plt.savefig(f"{OUT}/pocket_selection.png", dpi=150)
print("saved pocket_selection.png")

# Spearman over valid families (S vs error)
from scipy.stats import spearmanr
S = [r["S"] for r in res.values() if r["valid"]]
E = [r["test_mean"] for r in res.values() if r["valid"]]
rho, pval = spearmanr(S, E)
print(f"[antideriv] Spearman(S, testerror) over {len(S)} valid families: rho={rho:.3f}")

# ---------------- Figure 2: generalization across operators ----------------
pdes = {"antideriv": ("Antiderivative (linear)", "o"),
        "diffusion": ("Diffusion (linear)", "s"),
        "burgers":   ("Burgers' (nonlinear)", "^")}
pcol = {"antideriv": "#1f77b4", "diffusion": "#2ca02c", "burgers": "#d62728"}
fig2, ax2 = plt.subplots(figsize=(6.2, 4.4))
rho_txt = []
for pde, (lbl, mk) in pdes.items():
    rr = json.load(open(f"{OUT}/results_{pde}.json"))
    sx = [v["S"] for v in rr.values() if v["valid"]]
    ey = [v["test_mean"] for v in rr.values() if v["valid"]]
    es = [v["test_std"] for v in rr.values() if v["valid"]]
    order = np.argsort(sx); sx = np.array(sx)[order]; ey = np.array(ey)[order]; es = np.array(es)[order]
    ax2.errorbar(sx, ey, yerr=es, marker=mk, ms=9, capsize=3, lw=1.3, color=pcol[pde], label=lbl)
    rho_p, _ = spearmanr(sx, ey); rho_txt.append(f"{lbl.split()[0]}: $\\rho$={rho_p:.2f}")
    print(f"[{pde}] Spearman rho = {rho_p:.3f}")
ax2.set_xscale("log")
ax2.set_xlabel(r"pocket score  $S=(D\cdot T)^{1/2}$  (training-free, computed from inputs)")
ax2.set_ylabel(r"test relative $L_2$ error")
ax2.set_title("Pocket score predicts error across three solution operators")
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(title="  ".join(rho_txt), fontsize=9, title_fontsize=8, loc="lower left")
plt.tight_layout()
plt.savefig(f"{OUT}/pocket_generalization.png", dpi=150)
print("saved pocket_generalization.png")
