# Research log

**ShouldYouEvenNN — a notebook on when to continue.**

Entries are recorded in America/Bogota local dates and appended in chronological
order. This notebook preserves questions, observations, failures, and decisions.
The [protocol](research-protocol.md) defines the study; linked run records will
carry the experimental evidence. An unanswered question stays unanswered here.

## 2026-09-29 | Entry 001 | A clean bench

Recorder: Codex, research and development assistant.  
Kind: research setup, with a retrospective account of the audit and migration.  
Starting revision: `a346af2` (`Refactor to start`), branch
`codex/research-foundation`; the working tree was clean before this entry's changes.

The old machinery is preserved. The question remains open.

**Question and working hypothesis.** Can a small amount of training evidence tell
us when further work on a particular neural candidate is unlikely to produce a
worthwhile improvement over a classical baseline? We suspect that a simple stopping
rule can save compute while preserving most worthwhile candidates. That suspicion
has not been established. If the rule repeatedly discards eventual winners, or its
own overhead consumes the savings, the experiment will have challenged its purpose.

**Record.** The earlier audit exposed problems in candidate reconstruction,
uncertainty estimates, baseline comparisons, and result accounting. Its limitations
are summarized in the [prototype guide](../archive/thesis-prototype/ARCHIVE.md).
We chose to preserve that implementation and establish a smaller research
foundation. The user's chosen target is the decision to continue training a
candidate under a specified budget.

The [migration verification](migration-verification.md) records 1,394 matching
file hashes, two passing packaging tests under both editable and wheel
installation, and passing lint and formatting checks. These are previously
recorded engineering checks, not experiments rerun for this entry. They establish
preservation and packaging; they say nothing about the stopping rule's effectiveness.

Today we established [repository instructions](../AGENTS.md) to read and maintain
this notebook. No training experiment was run. The active package still has no
scientific implementation.

**Interpretation and decision.** We have a place to work and a record of what must
be questioned. We do not yet have evidence for a continuation policy. The next
study will begin with a fixed candidate procedure, a matched classical baseline,
one decision checkpoint, and one endpoint. Extra mechanisms must earn their place
through a measured benefit.

**Next test.** Specify the smallest controlled continuation experiment before
implementing it: dataset separation, model configurations, metric, budgets,
minimum worthwhile improvement, and acceptable missed-opportunity rate. Complete
the candidates that the simulated policy would stop so that eventual winners
remain observable. The immediate task is to make that protocol concrete; its
outcome is still unknown.

## 2026-09-29 | Entry 002 | The question has ancestors

Recorder: Codex, research and development assistant.  
Kind: completed targeted literature investigation; no training experiments.  
Starting revision: `a346af2`, branch `codex/research-foundation`; notebook,
repository instructions, and documentation links already had uncommitted changes.

The bibliography contains a challenge to the machinery we were preparing to build.

**Question and method.** Where does the continuation study belong, and what must
its simplest credible comparison be? We examined the archived thesis's background
and bibliography, then consulted primary papers, including selected method sections
on early discarding, Bayesian extrapolation, and learning-curve cross-validation.
This was a targeted reading, not a systematic review of every cited work or a
completed novelty search. The earlier project audit primarily assessed the
implementation and validity of its evidence.

**Observations.** [Mohr and van Rijn's survey](https://arxiv.org/abs/2201.12150)
explicitly includes deciding whether a learner at a given budget will beat a
reference. The broad question therefore has established predecessors.
[Domhan et al.](https://www.ijcai.org/Proceedings/15/Papers/487.pdf) model uncertainty
over extrapolations and the probability of beating an incumbent.
[LC-PFN](https://arxiv.org/html/2310.20447v1) approximates predictive distributions
using a model trained on synthetic curves from an explicit prior. Both connect
directly to the jurors' suggestion; their assumptions still require scrutiny.

[Egele et al.](https://arxiv.org/html/2404.04111v1), already cited in the thesis,
found little additional utility from several sophisticated discarding methods
over fixed-epoch screening in their studied settings. Their i-Epoch comparison
should be central to our evaluation; this is not a universal one-epoch guarantee.
[LCCV](https://arxiv.org/pdf/2111.13914) extrapolates performance as training-set
size grows and relies on a convexity assumption. That assumption cannot simply be
transferred to neural accuracy over epochs.
[McElfresh et al.](https://arxiv.org/abs/2305.02997) support examining dataset
properties, tuning, and the practical size of performance differences when
comparing neural networks with boosted trees.

**Interpretation and recommendation.** The promising study is a narrow evaluation
of whether a continuation rule saves compute while retaining candidates that would
meaningfully improve on a matched classical baseline. Referencing a different
baseline alone does not establish novelty. Fixed-epoch screening must be allowed
to win. Distributions should express uncertainty about a defined endpoint, and
their reliability should be evaluated separately from the quality of the stopping
decisions. These are recommendations from this reading; no scientific method has
been implemented or validated today.

**Next test.** Specify one checkpoint and endpoint, a meaningful improvement
threshold, and separate development and evaluation datasets. Compare always
finishing, simple fixed-epoch screening, and one justified continuation method.
Observe completed outcomes even for simulated rejections; measure saved compute,
missed worthwhile candidates, and the size of their missed improvements. Keep
preference parameters distinct from quantities estimated from data.
