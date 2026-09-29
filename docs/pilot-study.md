# Pilot 01: is early performance informative enough?

Status: proposed design, 2026-09-29. No experiments have run. This pilot develops
the [continuation protocol](research-protocol.md); it is not final evaluation.

## Question

Can performance at one early checkpoint identify neural candidates that will
meaningfully improve on a classical baseline at a fixed training endpoint?

The immediate output is evidence about simple screening, including its failures.
A new forecasting algorithm is not required to answer this first question.

## Proposed scope

Start with public, independent tabular binary-classification datasets, a fixed
portfolio of MLP configurations, and one gradient-boosted-tree baseline family.
Restricting the task is a pragmatic pilot choice, not a claim of generality.
Specify and budget baseline tuning explicitly before calling the comparator strong.
The motivation for including tuned trees comes from
[McElfresh et al.](https://arxiv.org/abs/2305.02997).

Use mean log loss as the proposed primary metric: lower is better, and it evaluates
predicted probabilities ([metric definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)).
Record accuracy as a secondary descriptive metric. Fix probability handling and
class order in the implementation; do not choose the primary metric after results.

Select datasets by a documented rule covering size, class balance, feature types,
and availability within the compute budget. Record source/version and exclusions.
Assign related datasets to the same development or final-evaluation group. All
datasets inspected during this pilot become development data. The later evaluation
collection must remain uninspected while the rule is developed.

## Observation and decision

Define each run by dataset, split, configuration, and training seed. Fit each
model's preprocessing on the training partition only. Use the same held-out rows
for candidate and baseline comparisons; model-specific preprocessing is permitted.

Train every candidate to an endpoint T, preserving the same training trajectory.
Record validation loss and cumulative elapsed time each epoch. Use one checkpoint
t < T for the primary pilot analysis. Operational failures and resource cutoffs
remain explicit incomplete outcomes, not losing models.

At t, calculate the early validation advantage:

`early_advantage = baseline_validation_loss - candidate_validation_loss_at_t`

The simplest rule continues when `early_advantage >= threshold`. It may use only
information available at t. For each development threshold, replay that decision
against the completed trajectory. A candidate is an observed worthwhile winner
when `baseline_validation_loss - candidate_validation_loss_at_T > delta`.
This pilot label concerns held-out validation performance, not proven population
superiority. Final test-set performance will be assessed only after policy and
model selection are frozen, under the parent protocol.

Compare always continuing, always stopping at t, and this threshold rule. When a
candidate is stopped, the declared fallback is the classical baseline. Keep this
per-candidate decision separate from choosing the best model across a portfolio.
If we later claim search-level gains, include an explicit fixed-epoch ranking and
promotion comparison such as [i-Epoch](https://arxiv.org/html/2404.04111v1).
The baseline-referenced threshold rule is an adaptation, not a reproduction of
that paper's algorithm.

## Deliverables

- An immutable split manifest and versioned run configuration.
- Complete learning curves with provenance, timing, and failure records.
- Early-versus-final advantage plots that identify late-improving candidates.
- A compute-savings versus missed-winner curve, with lost improvement magnitudes.
- Per-dataset and per-seed results; avoid treating epochs as independent samples.
- A short interpretation stating where simple screening fails and what, if
  anything, an uncertainty model would need to improve.

Report missed winners as a fraction of all observed worthwhile winners, including
the numerator and denominator. If there are no winners, report that fraction as
undefined; it is not evidence of safe screening. Show failures separately and do
not claim policy quality from only successfully completed runs without describing
that restriction. Account for baseline, preprocessing, validation, and decision
costs, distinguishing simulated deployment savings from full pilot collection cost.

## Parameter ledger to complete before collection

| Choice | How to establish it |
| --- | --- |
| Hardware and total collection budget | Local PC use authorized; [hardware and environment checked](local-environment.md). Set collection scale after a timing check; no numeric wall-time cap specified. |
| Dataset IDs, count, and splits | Published source and explicit eligibility rule; size within measured budget. |
| MLP portfolio and baseline tuning | Small documented search spaces with primary-source rationale; freeze before collection. |
| Endpoint T and checkpoint t | Engineering timing on development data plus a stated training horizon; no claim that a convenient epoch count is universally sufficient. |
| Number and identities of seeds | Budgeted repeatability and variability assessment; disclose limited precision in a small pilot. |
| Meaningful improvement delta | Practical preference in log-loss units; a development sensitivity analysis can inform, but cannot replace, the later declared objective. |
| Screening threshold | Explore only on development results, then freeze for independent evaluation. |
| Acceptable missed-winner rate | User/research objective; not inferred from a paper's default. Report the pilot trade-off before setting a confirmatory target. |

Every numeric setting must identify its role: measured quantity, literature-based
assumption, engineering constraint, or preference. Literature defaults are starting
points with applicable conditions, not universal constants.

## Execution order

1. Establish available compute and select a documented dataset shortlist.
2. Implement one auditable full-trajectory collector and run a development timing
   check. Estimate collection cost before scaling up.
3. Freeze the pilot configuration and complete its candidates and baselines.
4. Implement replay of the simple controls, verify timing and decision boundaries,
   and produce the descriptive analysis above.
5. Decide from the observed failures whether probability estimation merits a next
   experiment. A small pilot cannot establish a general missed-opportunity bound.
