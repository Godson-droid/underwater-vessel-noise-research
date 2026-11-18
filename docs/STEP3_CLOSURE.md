# Step 3 Closure — Formal Negative Finding

Status: CLOSED by Godson's decision, 2026-11-18. Stage 2 is not authorised and does not proceed under the Step 3 authorisation.

Source record: RESULTS_step3_stage01.md (this closure summarises it and records the decision; it does not replace it).

## The decision

Godson, 2026-11-18: Stage 0 passed; Stage 1 failed at its first predeclared test. The physical energy decomposition hypothesis, in its two minimal forms, is rejected for this programme. The work stopped exactly where the protocol required: neither estimator was run on the main I1 set, no quantile was searched, the spectral model was not tuned, and no threshold was reinterpreted after the failures were seen. A more complicated hybrid would be a new hypothesis requiring a separate proposal and approval.

## Decisive results

Stage 0, independent synthetic construction: PASSED. Oracle score 0.9989 against the construction's known labels. Hull and date disjointness verified (0 shared hulls, 0 shared dates), labels hand checked to 2.7e-09, baseline comparison recorded (frozen Phase 5 regressor scores 0.8854 overall on I1, reference only).

Stage 1, null test (550 pure background windows from held out sites, true share zero; pass requires mean at most 0.03 AND median at most 0.02): BOTH ESTIMATORS FAILED.

- S1 temporal envelope: mean 0.5747, median 0.6285. Rejected.
- S2 spectral family decomposition: mean 0.0999, median 0.0920. Rejected.
- Phase 5 regressor on the same null material: mean 0.0471, median 0.0340. Reference only; it also fails this stricter independent null test.

## What the experiment established

- S1 rejected: its calibration correction was learned from event contaminated windows, so it learned to discount energy that is not present in pure background windows, severely underestimating background energy and producing false vessel share predictions.
- S2 rejected: its background family, learned from six sites, could not represent the held out sites; the unexplained background energy was attributed to the added passage component. The unseen site hallucination is reproduced through a different estimation mechanism.
- Supported: unseen site background characterisation is the persistent difficulty among the strategies tested. The regressor, the temporal envelope and the spectral decomposition all over read pure backgrounds from held out sites. This does not prove the 15 x 25 summaries make decomposition fundamentally unidentifiable.

## What must not be inferred

- The 0.9989 oracle score demonstrates recoverability under this synthetic construction when the true background is supplied. It does not demonstrate that an estimator can recover that background, or that the same recovery is possible on real test windows.
- I1 is independently implemented and uses disjoint held out hulls and dates, but shares the underlying public data sources and synthetic energy additive assumptions. It is not independent real world validation.
- Because the null test failed, there are no valid Stage 1 overall or unseen site scores for S1 or S2. No performance comparison may be manufactured from results the protocol deliberately did not produce.

## Implementation incident (preserved deliberately)

The first S1 null run showed an apparent pass (mean 0.0132) caused by a mask shape bug in the prediction code (cell mask summed where a minute count was required, inflating the estimated background about 25x). It was caught on code review before any reliance, fixed, the stale predictions deleted, and the run repeated. Every reported number comes from the fixed code. Documenting the incident makes the evidence more credible, not less.

## Distinctions to keep explicit in the final write up

1. Stage 0 passed: a verified independent synthetic validation construction is now available.
2. Stage 1 failed: both minimal decomposition estimators failed the held out site null test.
3. Research conclusion: unseen site background characterisation remains a persistent estimation difficulty across the tested approaches.
4. Still unresolved: whether a more expressive estimator could separate background and passage energy from the summaries, and whether the same difficulty transfers to real test data.

## Standing facts

No candidate was produced, no submission was made, no slot was used (still 1 of 200). No inference was run on the real test set. The Phase 0 to 8B archive remains frozen and untouched.
