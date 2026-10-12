# Working rules
- Follow PLAN.md in order. For each step: say the goal in one line, do it, then PROVE it
  with evidence (printed shapes, metrics, saved figures), and tick the checkbox.
- Run everything yourself and read errors yourself; fix and retry up to 3 times before asking me.
- Write code as .py scripts (not notebooks). Log with `2>&1 | tee logs/<step>.log`.
- Save every plot to outputs/ and look at it yourself to check it is sane.
- Smoke-test on a tiny subset (--max-samples 50) before any full run.
- After each step add 3 lines to LEARNING.md: what we did, why, what a wrong result would look like.
- Only stop to ask me at decision points or checkpoints marked in PLAN.md.
- Keep RESULTS.md updated with current metrics.
- train.py must take a --data-dir argument so the same script runs locally and on Kaggle.
