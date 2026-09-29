# Working on ShouldYouEvenNN

The research question is: **when should training this neural candidate continue?**
Favor the smallest experiment that can answer it. Keep historical material in
`archive/` unchanged unless the user specifically requests otherwise. Use
`docs/research-protocol.md` for scientific scope and `docs/development.md` for
environment and verification commands.

## Keep the research notebook alive

The shared notebook is `docs/research-log.md`. Create it if absent.

- Before substantive work, read the latest entries and any earlier entry relevant
  to the task. Carry forward unresolved questions and failed approaches.
- Append an entry after a meaningful investigation, experiment, methodological
  decision, implementation milestone, or failure. Record consequential user
  decisions too. Routine formatting and conversational acknowledgments do not
  require an entry.
- For long experiments, record the question and planned test before execution;
  append the result afterward. Clearly label proposed, running, completed, failed,
  or inconclusive work. A plan is never evidence that an experiment happened.
- Keep entries chronological with a local date (America/Bogota), a sequential entry
  number, and a short descriptive title. Do not invent precise timestamps.
- Preserve prior observations. Correct an error with a dated amendment referencing
  the affected entry rather than silently rewriting the record. Identify any
  retrospective entry and the records used to reconstruct it.

## Voice and substance

Write like a scientist keeping a field notebook: lucid, curious, restrained, and
specific. A little atmosphere is welcome; invented drama is not. Use connected
prose and first-person plural for work actually done together. Identify the
recorder as an assistant; do not invent human experiences, emotions, or lab events.
Summarize scientific rationale, not a transcript of internal deliberation.

Each substantial entry should make these points easy to find; shorten or omit
sections that do not apply:

1. **Question / hypothesis:** what we want to learn and what could disprove it.
2. **Method:** what actually ran or changed. For experiments, identify the code
   revision and dirty state, configuration, data/split, seeds, budget, and environment
   through links to the run record rather than duplicating it.
3. **Observations:** measured results, units, sample sizes, failures, and evidence
   links. Distinguish fresh checks from previously reported findings.
4. **Interpretation / decision:** what the evidence supports, what it does not,
   and the resulting decision or change in direction.
5. **Next test:** the smallest useful next action and unresolved uncertainty.

Never fabricate measurements, citations, successful checks, or certainty. Separate
observations from hypotheses and preferences. Report negative results and missing
outcomes honestly. Keep secrets and sensitive row-level data out of the notebook.

The log explains the research journey; protocols specify experiments and run
artifacts hold detailed evidence. Link between them. Keep code tests appropriate
to the change; documentation-only edits need link and diff checks, not training.
