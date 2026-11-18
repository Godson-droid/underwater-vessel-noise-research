"""Phase 3 prior diagnosis: what the two public scores pin down.

Known, on the (unknown) public split of ~30% of the 1,500 test windows:
  constant 0.1 predictor : RMSE_b = 1 - 0.753867825 = 0.246132175
  our v2 submission      : RMSE_v = 1 - 0.687868143 = 0.312131857
Our predictions p are known exactly (submission file); their moments
below are computed over all 1,500 rows and used as estimates of the
public-split moments (the split is a fixed subset; if it is effectively
random, subset moments differ from full-file moments by sampling noise
of order sd/sqrt(450) ~ 0.01 on the mean — flagged where it matters).

Equations, with s the true shares on the public split, mu = E[s],
var = Var(s):
  (1) E[(s-0.1)^2] = var + (mu-0.1)^2            = RMSE_b^2
  (2) E[(s-p)^2]   = var + Var(p) - 2 Cov(s,p)
                     + (mu - E[p])^2             = RMSE_v^2
Two equations, three unknowns (mu, var, Cov): the pair does NOT pin
down mu and var separately. It pins down var + (mu-0.1)^2 exactly, and,
for each assumed mu, the Cov (hence correlation rho and the optimal
linear shrinkage of p) implied by equation (2).
"""
import numpy as np
import pandas as pd
from common import DATA

sub = pd.read_csv(DATA.parent / "submissions" / "submission_sim_v2.csv")
p = sub["share_added_vessel"].to_numpy(dtype=np.float64)
Ep, Vp = p.mean(), p.var()
print(f"v2 predictions (all 1500): mean {Ep:.4f}, var {Vp:.4f}, sd {p.std():.4f}")

RMSE_b, RMSE_v = 1 - 0.753867825, 1 - 0.687868143
print(f"RMSE_b^2 = {RMSE_b**2:.6f}   RMSE_v^2 = {RMSE_v**2:.6f}")
print(f"From (1): var = {RMSE_b**2:.6f} - (mu-0.1)^2, so var <= {RMSE_b**2:.6f} "
      f"(sd <= {RMSE_b:.4f}) and mu in [0, {0.1 + RMSE_b:.4f}]")

print("\nmu     var     sd     implied Cov   implied rho   opt slope b   "
      "opt intercept a   best shrunk RMSE")
for mu in (0.10, 0.13, 0.17, 0.20, 0.25, 0.30):
    var = RMSE_b**2 - (mu - 0.1) ** 2
    if var <= 0:
        continue
    cov = (var + Vp + (mu - Ep) ** 2 - RMSE_v**2) / 2.0
    rho = cov / np.sqrt(var * Vp)
    b = cov / Vp
    a = mu - b * Ep
    best = np.sqrt(max(var - cov**2 / Vp, 0.0))
    print(f"{mu:.2f}  {var:.4f}  {np.sqrt(var):.4f}   {cov:+.4f}      "
          f"{rho:+.3f}        {b:+.3f}        {a:+.3f}          {best:.4f}")

print("\nReading: the implied correlation between the v2 predictions and")
print("the truth depends on the unknown mu: rho = 0.63 at mu = 0.10,")
print("0.51 at mu = 0.13, 0.37 at mu = 0.17, 0.26 at mu = 0.20, ~0 at")
print("mu = 0.25. For the plausible range (the benchmark cross-check in")
print("RESULTS_sim_v2.md favours mu near 0.13-0.20), rho is roughly")
print("0.26-0.51 and even the OPTIMAL linear shrinkage of the v2")
print("predictions caps at RMSE 0.21-0.22 (score 0.78-0.79) — above the")
print("0.7539 benchmark but nowhere near the top of the board. Shrinkage")
print("of a weak-signal model is a floor defence, not a route to 0.93+;")
print("the signal itself (background recovery) has to change.")
print("\nAlso pinned down exactly by (1) alone: the best possible constant")
print("predictor scores RMSE = sd(s) >= sqrt(RMSE_b^2 - (mu-0.1)^2); if")
print("mu = 0.17 (our synthetic prior mean), sd = "
      f"{np.sqrt(RMSE_b**2 - 0.07**2):.4f} and the best constant (0.17) "
      f"would score {1 - np.sqrt(RMSE_b**2 - 0.07**2):.4f}.")
