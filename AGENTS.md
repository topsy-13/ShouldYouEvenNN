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
specific. Keep notes very brief: aim for 3-5 sentences and at most 100 words per
entry, excluding its heading. Use one compact paragraph or a few short bullets;
do not expand entries into a multi-section report. Identify the recorder briefly
as an assistant. Avoid atmospheric introductions and procedural narration.

Keep only what matters for understanding the research or deciding what to do next:

- The question or consequential change, only if needed for context.
- The key finding or decision, with essential evidence and any limitation that
  changes its interpretation. Distinguish new results from earlier observations.
- The next useful test or unresolved blocker, when applicable.

Link to protocols, run records, or reports for methods, commands, dependency lists,
installation troubleshooting, and detailed results. Experiment run records must
retain the revision and dirty state, configuration, data/split, seeds, budget, and
environment; do not duplicate them in the log. Omit routine checks, repeated
background, and intermediate attempts unless they change the conclusion or next
action. Preserve relevant negative findings and missing outcomes even when brief.

Never fabricate measurements, citations, successful checks, or certainty. Separate
observations from hypotheses and preferences. Report negative results and missing
outcomes honestly. Keep secrets and sensitive row-level data out of the notebook.

The log explains the research journey; protocols specify experiments and run
artifacts hold detailed evidence. Link between them. Keep code tests appropriate
to the change; documentation-only edits need link and diff checks, not training.
