# ARCHIVED — superseded by data/processed/evidence_corpus.jsonl

**Do not use these files for retrieval.** They predate the locked
evidence-unit design and, critically, **have no golden-set leakage guard.**


## What generated these files

`build_retrieval_corpus.py` (also archived here), given
`data/processed/applesupport_threads.jsonl` as input.

## What its eligibility definition was

A thread was "eligible" (`retrieval_corpus.jsonl`) if it was coherent, had
both a customer and a brand turn, and was NOT `dm_deflection_only` — a
whole-thread word-count + keyword heuristic (a thread is excluded only if
*every* brand turn is short AND matches a DM/contact-us pattern). This is
a much more permissive bar than the canonical definition: it asks "is this
thread NOT purely deflection," not "does this thread contain an actual,
extractable instruction."

## Why it's archived, not just "a different definition"

1. **No golden/dev exclusion mechanism existed in `build_retrieval_corpus.py`
   at all** — no `--excluded-root-ids-file`, no exclusion logic of any
   kind. Checked directly against the real 180 golden root_ids: **all 180
   are present in `retrieval_corpus_candidates.jsonl`, and 147 of them
   (82%) are present in the "eligible" `retrieval_corpus.jsonl`.** Using
   this file for retrieval would have contaminated the majority of the
   golden evaluation set.
2. Its permissive, whole-thread heuristic let a real, non-trivial number of
   templated non-answers through as "eligible" (see the corpus's own
   README for the ~28% padded-deflection finding from manual inspection).
3. Nothing else in the repo reads these files — confirmed by inspection,
   not assumption. They are dead weight, not a dependency.

## What replaces it

`data/processed/evidence_corpus.jsonl`, built by
`scripts/build_evidence_corpus.py` from the same `applesupport_threads.jsonl`
input, using the calibrated `reply_classification.py` classifier
(sentence-level, not whole-thread) and a real golden-root exclusion file
(`scripts/excluded_root_ids_golden180.txt`, joined against real message
data — all 180 golden roots resolved and excluded, verified with zero
overlap).

See the repo's decision log for the full reasoning.
