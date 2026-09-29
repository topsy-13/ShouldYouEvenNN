# Continuation study: protocol outline

Status: design outline, not an implemented or preregistered experiment.

## Question and hypothesis

Given a particular neural candidate's early training history, a specified classical
baseline, and a fixed endpoint budget, can a stopping rule save meaningful compute
while keeping the rate of incorrectly stopped worthwhile candidates acceptable?

The claim is conditional on the candidate, training procedure, dataset population,
and budget evaluated. It does not establish that all neural networks are unsuitable
for a dataset.

## Minimal experiment

Begin with a modest fixed MLP configuration set, a specified classical baseline,
full training data throughout, one predetermined decision checkpoint, and one
fixed endpoint. Compare the proposed rule against always continuing, always
stopping, and a rule based on current validation performance.

The dataset collection, model configurations, metric, budgets, checkpoint, minimum
worthwhile improvement, and acceptable missed-opportunity rate remain to be
specified before running the study. Historical constants are not adopted by default.

## Evaluation separation

Separate datasets used to develop and calibrate the rule from final evaluation
datasets; keep related datasets together. Within a dataset, define training,
validation, and test partitions explicitly. Fit preprocessing on training data.
Compare baseline and candidate under a matched protocol and evaluation partition.
Keep the test set out of model selection, calibration, and stopping decisions.

Forecast a specified endpoint of the same continued candidate. Preserve its model,
optimizer, and other training state; rebuilding a different model is not a matched
continuation. Keep training variability, held-out sample uncertainty, and forecast
uncertainty distinct. Repeated checkpoints would require a revised evaluation of
the sequential policy, not a reuse of single-checkpoint guarantees.

## Counterfactuals and missed opportunities

For evaluation, complete every candidate to the endpoint, including candidates the
rule would stop. These completed continuations reveal otherwise unobserved missed
opportunities. Report the fraction of worthwhile candidates incorrectly stopped,
performance lost relative to always continuing, and uncertainty calibration.

Predefine how worthwhile improvement, ties, failures, and missing outcomes are
handled. Missing results must never be converted to measured failures. Report
variation across seeds and datasets; correlated batch observations are not
independent experimental replicates.

## Compute accounting

Choose and declare the budget unit before running. Log actual elapsed time and
training work separately. Include validation, forecasting, decision overhead, and
baseline costs in end-to-end accounting, stating any shared costs explicitly.

Distinguish the policy's simulated or measured operational cost from the extra
counterfactual cost incurred only to evaluate it. Compare against matched controls
on the same hardware and protocol. Do not silently grant extra training after the
budget expires.

## Evidence required before an effectiveness claim

Freeze the protocol, configurations, calibration procedure, and acceptance criteria
before final evaluation. Report compute savings alongside missed-opportunity rates
and performance loss, with dataset-level uncertainty. Retain failed runs and
document exclusions. Add complexity only when an ablation supports its value.
