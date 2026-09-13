#!/usr/bin/env python3
"""
extract_examples.py

Pulls the 180-example golden set out of golden_labeling_tool.html and writes

a model-facing input file that contains ONLY example_id + raw_text.

This is a deliberate physical separation: provenance (sampling_stratum_hint,
boundary_candidate_design, overlap_pair, nearest_corpus_sim,
nearest_corpus_root_id, month, is_launch_window, tweet_id) never leaves this
script. generate_prelabels.py only ever reads the output of this script, so
it structurally cannot see hidden metadata -- there's nothing to accidentally
leak downstream.

Usage:
    python3 extract_examples.py golden_labeling_tool.html examples_for_prelabeling.json
"""
import json
import re
import sys


def extract(html_path: str) -> list[dict]:
    html = open(html_path, "r", encoding="utf-8").read()
    m = re.search(
        r'<script type="application/json" id="example-data">(.*?)</script>',
        html,
        re.S,
    )
    if not m:
        raise SystemExit(f"Could not find #example-data block in {html_path}")
    examples = json.loads(m.group(1))

    # Strip everything except example_id + raw_text. This is the ONLY
    # information that should ever reach the pre-labeling model.
    stripped = [
        {"example_id": ex["example_id"], "raw_text": ex["raw_text"]}
        for ex in examples
    ]
    return stripped


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <golden_labeling_tool.html> <output.json>")
        sys.exit(1)

    html_path, out_path = sys.argv[1], sys.argv[2]
    stripped = extract(html_path)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stripped, f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(stripped)} examples -> {out_path}")
    print("Fields present:", sorted(stripped[0].keys()) if stripped else "none")


if __name__ == "__main__":
    main()
