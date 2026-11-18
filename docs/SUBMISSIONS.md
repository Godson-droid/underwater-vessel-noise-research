# Zindi Underwater Vessel Noise — submission log

Competition deadline: 13 Dec 2026. Budget: 2 per day, 200 total. Final: choose 2 submissions for the private leaderboard before the close.

| # | Date (WAT) | File | Submission ID | Public score | Rank | Notes |
|---|---|---|---|---|---|---|
| 1 | 2026-11-16 | submission_sim_v2.csv | tsB9FZT6 | 0.687868143 | 600 | Simulator v2 model. BELOW the constant baseline benchmark 0.753867825. Synthetic validation was 0.908–0.911, so the simulator to real gap is large: the model overpredicts added energy on real windows (test predictions mean 0.330). Top of board: 0.994250562 (melihk, 37 submissions), which suggests the generation process is almost exactly recoverable, plausibly by matching test backgrounds against the released South African data. |

## Phase 3 direction (from the submission 1 diagnostic)
1. Test whether test window backgrounds match windows in the released SA acoustics; if they do, recover the added component by subtraction in linear energy space.
2. A heavily shrunk variant of the v2 predictions is a cheap floor test: constant 0.1 scores the benchmark 0.7539, so shrinkage toward low values should clear it.
