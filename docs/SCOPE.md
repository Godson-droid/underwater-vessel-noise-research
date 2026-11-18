# Zindi Underwater Vessel Noise — Scope of Work

Owner: Godson (Zindi account Godson22)
Competition: Underwater Vessel Noise Challenge by the French Embassy in South Africa
Deadline: 13 December 2026
Prize pool: $5,000. First $1,500 plus a ChangeNOW Summit 2027 invitation worth $2,500, second $1,000, third $500, plus two best writeup prizes of $1,000 each.

## The task, in plain terms

Each test case is one window of underwater sound: 15 one minute observations across 25 frequency bands, given as sound energy summaries. There is no raw audio. For every window we predict one number between 0 and 1: what fraction of the window's total sound energy came from a vessel passage that the organisers deliberately added to a real recorded background.

The score is 1 minus the root mean square error on that fraction. Only that fraction counts. The public leaderboard reflects about 30 percent of the test windows; final rankings use the remaining 70 percent. Before the close we choose 2 of our submissions to count for the final ranking.

## Rules that bind the work

- Submission file: one row per test window, columns ID, share_added_vessel, score_added_vessel, all values finite and within 0 to 1.
- Budget: 2 submissions per day, 200 in total. Every submission is a measurement and is spent deliberately.
- Test inputs are for inference only. Nothing may be fitted, fine tuned, calibrated or pseudo labelled from them.
- Open source tools only, no AutoML. Code must be original, seeded and reproducible; the top finishers hand in code that must reproduce the score.
- External data is allowed when it is public and openly licensed, with sources declared. The organisers state the added passages come from the VTUAD v1.3.2 dataset (CC BY 4.0) converted with the starter notebook's own function, then combined with measured South African backgrounds, with absolute gain discarded and the passage's trajectory, channel colour and energy varied first.
- A research writeup is expected as part of a serious entry: pressure maps, holdout evidence on real South African data, and comparison against a simple acoustic baseline.

## Where we stand, measured

- Benchmark (the organisers' constant prediction of 0.1 everywhere): 0.7539.
- Our first submission (16 November, simulator model v2): 0.6879, rank 600 of the public board. Below benchmark. Its predictions on real windows average 0.330, too high.
- Top of the public board: 0.9943. Near perfect recovery exists, so the information needed is present in the data. The gap is in our method, not in the task.

## Work done so far

Phase 0, audit (complete). Every competition file was downloaded and measured directly. The format question was settled from the files themselves: band energy regression, not vessel type classification. Baselines were built and validated mechanically. A null test proved the simple low band excess statistic is not viable on its own: on pure background windows with a true answer of 0 it reads about 0.34 on average.

Phase 1, simulator v1 (complete). We built a training simulator: real background windows drawn from the released South African acoustics (87,653 clean 15 minute windows across 10 sites), vessel passages from VTUAD v1.3.2 converted with the organisers' own function, mixed in exact known proportions. A gradient boosting model on 549 acoustic features scored 0.905 to 0.917 on held out synthetic windows, against 0.75 for the constant baseline on the same windows.

Phase 2, simulator v2 (complete, submitted). Longer and more realistic passage events, wider variation, more ambient references, and a training fix so early stopping actually engages. It beat v1 on every new validation set, including a clear gain on long events (0.898 against 0.869). Submitted as our first entry. On the real leaderboard it scored 0.6879: the simulator still does not match how the organisers built the real test windows, and the synthetic scores did not transfer.

## Phase 3, the current investigation

Hypothesis: the real backgrounds inside the test windows are drawn from the same released South African recordings we already hold. If a test window's background can be found in the release, the added vessel energy is what remains after subtraction, and the fraction can be recovered directly instead of guessed by a model. A near perfect top score is consistent with this route.

The investigation, now running:
1. Build a matcher that searches the released backgrounds for each test window's background, using the high frequency bands and quiet minutes where added vessel energy is weakest, and requiring the candidate background to sit at or below the test window in energy, cell by cell, as a true background must.
2. Prove or kill the idea where the answer is known: synthetic windows built the organisers' way, background hidden from the matcher, recovery error measured across all share levels.
3. Weaker fallbacks if exact matching fails: background estimation from the same station and period, and from the window's own quiet minutes and bands.
4. A written verdict with numbers, whichever way it lands.

## The quality gate (set by Godson, 16 November 2026)

- The next submission must first prove a score of 0.93 or above on known answer validation that mirrors the organisers' process.
- A second, independent check (a different matching metric or a disjoint validation split) must agree at the same standard. One method agreeing with itself is not verification.
- Per window confidence must be reported, not just an average: how many windows resolve cleanly, how many are ambiguous, and what the method does with the ambiguous ones.
- Anything below the gate stays in the workshop and gets improved. It is not submitted on hope.
- Submission slots are spent only on files that carry this evidence, and the final 2 private leaderboard picks are chosen deliberately before the close, not by default.

## Honest risks and unknowns

- The organisers may have used backgrounds withheld from the release (held out stations or periods). If so, exact matching fails and Phase 3 falls back to statistical background estimation, with a lower ceiling.
- Their passage conversion may differ from the notebook function in ways we cannot see. The first submission's result says some mismatch exists; Phase 3 is designed to locate it.
- A proven 0.93 on synthetic validation does not guarantee 0.93 on the real board. It is the strongest evidence obtainable before spending a slot, and it is the standard Godson set.

## Timeline

- Now to late November: Phase 3 investigation and, if the gate passes, the evidence backed submission.
- Late November to early December: iterate on whichever route the evidence supports; build the writeup alongside, not after.
- Before 13 December: final submissions chosen, code packaged clean and reproducible for the code review, writeup completed.
