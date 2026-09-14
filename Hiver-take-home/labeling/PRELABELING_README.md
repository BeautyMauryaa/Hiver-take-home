# Model pre-labels vs. human golden labels — workflow

Three artifacts, kept structurally separate:

| File | Produced by | Contains |
|---|---|---|
| `golden_labeling_tool.html` | (unchanged, approved version) | The blind human labeling UI — no model suggestions, no changes from the version you approved before assisted-labeling was ever added. |
| `model_prelabels_v1.jsonl` | `generate_prelabels.py` (offline, run whenever) | Model's independent guess at the 5 fields, from `raw_text` only. |
| `golden_set_v1.jsonl` | The HTML tool's export button, after a human labels all 180 | The actual ground truth. Human labels only. |

## Steps

1. **Extract model-facing input** (strips provenance, keeps only `example_id` + `raw_text`):
   ```
   python3 extract_examples.py golden_labeling_tool.html examples_for_prelabeling.json
   ```

2. **Generate pre-labels offline**, any time, independent of human labeling:
   ```
   pip install anthropic
   export ANTHROPIC_API_KEY=sk-ant-...
   python3 generate_prelabels.py examples_for_prelabeling.json model_prelabels_v1.jsonl
   ```
   This never touches `golden_labeling_tool.html` and never sees `provenance`.

3. **Human labels the 180 examples blind**, using `golden_labeling_tool.html` exactly as approved — raw tweet text and the taxonomy/precedence/escalation cheat sheet only. No model output is shown anywhere in this UI. Export produces `golden_set_v1.jsonl`.

4. **Compare, after both passes are done** — join `golden_set_v1.jsonl` and `model_prelabels_v1.jsonl` on `example_id` for your agreement analysis / failure analysis section. This join should happen in your eval harness, not before or during labeling.

## Why this is separated the way it is

- `extract_examples.py` physically cannot leak provenance downstream — the file it writes never contains it, so there's nothing to accidentally forward to the model.
- `generate_prelabels.py` is a plain offline script using your own `ANTHROPIC_API_KEY` — no key is embedded in, or called from, anything that runs in the annotator's browser.
- The HTML tool has zero code path that reads `model_prelabels_v1.jsonl`. It's not "hidden but present" — it's simply never loaded into that page.
