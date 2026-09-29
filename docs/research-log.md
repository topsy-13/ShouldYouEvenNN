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

## 2026-09-29 | Entry 003 | First observe the trajectories

Recorder: Codex, research and development assistant.

Kind: pilot design proposed; collection and implementation not started.

Starting revision: `4a4753b`; the working tree was clean before this entry's changes.

We will need to see the endings before judging the early exits.

**Question and method.** The user accepted the literature framing as a starting
point and asked how to begin. We reviewed the existing protocol and drafted
[Pilot 01](pilot-study.md): full candidate trajectories followed by retrospective
screening. The draft proposes binary tabular classification, MLP candidates, a
budgeted boosted-tree comparator, and log loss as the primary measure. These are
scoping recommendations, not experimental findings or user-selected settings.

**Observations.** The active package still contains only its foundation. Dataset
selection, model settings, timing, and resource allocation remain open. We asked
for the available machine and compute budget. The draft distinguishes measured
parameters from preferences, assumptions, and resource constraints; no inherited
prototype constants were adopted. It also distinguishes a baseline-referenced
candidate rule from the search-level i-Epoch comparison discussed in Entry 002.

**Interpretation and next test.** Start with a documented dataset shortlist and
one auditable trajectory collector, then a development timing check to size the
pilot. Evaluate simple controls before building an uncertainty model. A pilot
with no worthwhile winners cannot demonstrate safe rejection. No dataset was
downloaded, model trained, or effectiveness claim established in this entry.

## 2026-09-29 | Entry 004 | The instrument is available

Recorder: Codex, research and development assistant.

Kind: completed local hardware and dependency assessment; no research training.

Starting revision: `4a4753b`; Entry 003 and the proposed pilot documentation were
uncommitted when this assessment began.

**Question and authorization.** The user authorized use of their PC and requested
an assessment of its components and installed dependencies. No numeric runtime
limit was supplied. Can the existing setup support the first pilot?

**Method and observations.** Read-only hardware and environment inspection found
a Ryzen 7 7700X, 16 logical processors, 31.12 GiB usable RAM, an RTX 4060 with 8 GiB
VRAM, and approximately 269.6 GiB free disk. Available RAM was 10.61 GiB at the
instant measured. CIM access failed; registry and Windows memory API checks
provided the CPU and memory information instead.

The existing AutoML environment has Python 3.11.11 and the required categories of
scientific libraries. Imports and pip dependency checks passed. A tiny synthetic
GPU multiplication and gradient computation passed under PyTorch 2.6.0+cu126.
This is an execution check, not a training experiment. The project's isolated
`.venv` contains development tools but no scientific stack. Exact inspected
versions and limitations are in the [environment assessment](local-environment.md).

**Interpretation and next test.** The PC is a credible platform for the proposed
pilot; runtime remains unmeasured. Use the working environment as a reference when
declaring project dependencies, keep GPU jobs serial initially, and profile one
development trajectory before setting collection size. No packages were changed.

## 2026-09-29 | Entry 005 | A separate research environment

Recorder: Codex, research and development assistant.

Kind: environment setup, started at the user's request.

Starting revision: `4a4753b`; pilot and environment-assessment documentation
already had uncommitted changes.

**Question and method.** Can we reproduce the working scientific stack in an
isolated environment declared by this repository? The user requested a YAML file
and a new environment. We added [environment.yml](../environment.yml), pinning
Python 3.11.11, the direct scientific versions observed in AutoML, and the project's
development tools. The NVIDIA build follows the CUDA 12.6 option in
[PyTorch's installation reference](https://pytorch.org/get-started/previous-versions/).

**Setup record.** Creation of `shouldyouevennn-research` began through Conda,
using conda-forge for Python and pip for scientific packages. A shell-profile
invocation failed before creation; invoking Conda's executable directly without
the login profile proceeded. Existing environments were not selected for updates.
This YAML pins direct requirements; it is not a complete transitive lock.

**Verification plan.** Install the active package, check imports and dependency
consistency, exercise a tiny GPU forward/backward calculation, and run the active
packaging tests and lint checks. Environment creation is not a research experiment.

**Installation adjustment.** Conda created Python successfully, but its buffered
pip step spent several minutes downloading the 2.5 GB PyTorch 2.6 wheel. We
interrupted that installer to inspect its output. A complete local cached PyTorch
2.12.0+cu130 wheel matches Python 3.11 and Windows; its SHA256 matches the official
PyTorch CUDA 13.0 index:
`00be49dbbe70a96fa6fd311e5e9cc7afb0f6e14730ce0fb9fd2bab22c98bfc3e`.
We revised the YAML to this version and resumed installation with that verified
wheel. This is an engineering version choice to reuse an available compatible
binary, not a scientific finding. GPU verification must be repeated for this new
version. Local installer artifacts are ignored under
`artifacts/runs/environment-setup/`.

**Completed verification.** Installation succeeded in the new named environment,
including the editable active package. Python, pip, and all direct YAML pins match
the installed versions. Scientific imports, GPU forward/backward execution on
the RTX 4060 with PyTorch 2.12.0+cu130, and `pip check` passed. Both packaging tests
passed, including the external-working-directory import test; Ruff lint and
format checks passed. Resolved package versions were saved with the ignored local
setup artifacts. No research dataset was downloaded or experiment launched.

## 2026-09-29 | Entry 006 | Shorter notes

Recorder: Codex, assistant. At the user's request, [logging guidance](../AGENTS.md)
now limits entries to the essential finding, decision, and next step, normally
3-5 sentences and no more than 100 words. Supporting details belong in linked
reports and run records. Earlier entries remain preserved.
