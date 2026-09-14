#!/usr/bin/env python3
"""
generate_prelabels.py

Offline, non-interactive step that produces model_prelabels_v1.jsonl.

Design constraints (do not relax these without re-reading the methodology
note this script was written against):
  - Input is examples_for_prelabeling.json, which contains ONLY
    {example_id, raw_text} per example (see extract_examples.py). This
    script has no access to provenance/sampling metadata because that
    file was never given it -- not because of a runtime filter.
  - The model is prompted with raw_text plus the locked taxonomy v1.1 /
    precedence rules / escalation policy v1.1 text (mirrored below,
    verbatim from the cheat sheet in golden_labeling_tool.html) and
    nothing else.
  - Output is written to its own file, separate from golden_set_v1.jsonl.
    It is never merged into the human labels. It exists so that, AFTER
    the independent human pass is complete, you can join the two files
    on example_id for an agreement/analysis step.
  - This script must be run BEFORE or independently of the human labeling
    pass, and its output must not be shown to the annotator during
    labeling (the labeling tool itself has no code path that reads this
    file, by construction).

Requires:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

Usage:
    python3 generate_prelabels.py examples_for_prelabeling.json model_prelabels_v1.jsonl
"""
import json
import sys
import time
from datetime import datetime, timezone

try:
    import anthropic
except ImportError:
    print("Missing dependency. Run: pip install anthropic", file=sys.stderr)
    sys.exit(1)

MODEL = "claude-sonnet-4-6"

VALID_INTENTS = {
    "Battery/Power", "Software/OS Bug", "Apple ID/Account Security",
    "How-to/Feature Guidance", "Backup/Data Recovery", "Billing/Payment",
    "App/Service Issue", "Repair/Physical Damage", "Connectivity", "Other/Ambiguous",
}
VALID_REASONS = {
    "ACCOUNT_SECURITY", "DEVICE_LOST_STOLEN", "BILLING_DISPUTE",
    "PHYSICAL_INSPECTION_REQUIRED", "PRIVATE_ACCOUNT_DATA_REQUIRED", "NON_ENGLISH_UNSUPPORTED",
}

# Verbatim (content-wise) from the human cheat sheet in golden_labeling_tool.html.
# Do not add anything about sampling, retrieval, or provenance here.
SYSTEM_PROMPT = """You are pre-labeling a customer tweet directed at @AppleSupport for a golden evaluation set. You will be given ONLY the raw message text. You have no other context -- no sampling metadata, no retrieval results, no classifier confidence. Judge the message on its own.

Decide these five fields:

1. is_support_request (true/false): true if this is an actual support request/complaint/question directed at Apple support. false if it's noise, off-topic, or unrelated chatter that merely mentions Apple/@AppleSupport.

2. intent (one of the following, or null if is_support_request is false): "Battery/Power", "Software/OS Bug", "Apple ID/Account Security", "How-to/Feature Guidance", "Backup/Data Recovery", "Billing/Payment", "App/Service Issue", "Repair/Physical Damage", "Connectivity", "Other/Ambiguous".

Precedence rules for overlapping cases:
- Battery vs. Bug: battery/charging/drain/overheat language -> Battery/Power, even if blamed on an update.
- Connectivity vs. Bug: WiFi/Bluetooth/cellular toggling itself -> Connectivity, even if blamed on an update.
- How-to vs. malfunction: "how do I fix X" where X is broken -> the malfunction category, never How-to. How-to is only for a working device/feature.
- App/Service vs. Bug: a specifically named app/service -> App/Service Issue. Unnamed, general device/OS symptoms -> Software/OS Bug.
- Apple ID vs. How-to: only an actual access failure (locked out, compromised, can't authenticate) -> Apple ID/Account Security. A working account with an informational question -> How-to.
- Billing vs. Battery ("charge"): money/transaction/refund context -> Billing/Payment. Physical charging cable/port/battery-level context -> Battery/Power.
- Multi-issue messages: pick the customer's stated main problem; if none stated, pick the most consequential/immediate need. No multi-intent label exists.

3. should_escalate (true/false) -- Flow 1, ground-truth only. Check in this order: account security/access failure -> lost or stolen device -> billing dispute or refund -> physical damage/repair -> non-English (input or requested) -> requires private account/order lookup. First match sets should_escalate=true with that reason as primary. If none apply, should_escalate=false by policy default -- assume adequate historical evidence exists; do not reason about retrieval quality or classifier confidence, that layer does not exist here. Hostile/abusive tone alone never escalates on its own.

4. escalation_reason (one of "ACCOUNT_SECURITY", "DEVICE_LOST_STOLEN", "BILLING_DISPUTE", "PHYSICAL_INSPECTION_REQUIRED", "PRIVATE_ACCOUNT_DATA_REQUIRED", "NON_ENGLISH_UNSUPPORTED", or null if should_escalate is false). Never use NO_GROUNDING_EVIDENCE or LOW_CLASSIFICATION_CONFIDENCE -- those are runtime-only codes.

5. boundary_case (true/false): true only if you seriously considered two or more taxonomy intents as plausible primary labels before applying the precedence rules above.

Respond with ONLY a single JSON object, no prose, no markdown fences, in exactly this shape:
{"is_support_request": true, "intent": "Battery/Power", "should_escalate": false, "escalation_reason": null, "boundary_case": false}"""


def sanitize(raw: dict) -> dict:
    out = {
        "is_support_request": raw.get("is_support_request") if isinstance(raw.get("is_support_request"), bool) else None,
        "intent": None,
        "should_escalate": raw.get("should_escalate") if isinstance(raw.get("should_escalate"), bool) else None,
        "escalation_reason": None,
        "boundary_case": raw.get("boundary_case") if isinstance(raw.get("boundary_case"), bool) else None,
    }
    if out["is_support_request"] is True and raw.get("intent") in VALID_INTENTS:
        out["intent"] = raw["intent"]
    if out["should_escalate"] is True and raw.get("escalation_reason") in VALID_REASONS:
        out["escalation_reason"] = raw["escalation_reason"]
    return out


def prelabel_one(client: "anthropic.Anthropic", raw_text: str) -> dict:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": raw_text}],  # ONLY raw_text, nothing else
    )
    text = "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")
    cleaned = text.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(cleaned)
    return sanitize(parsed)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <examples_for_prelabeling.json> <model_prelabels_v1.jsonl>")
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]
    examples = json.load(open(in_path, encoding="utf-8"))

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    n_ok, n_err = 0, 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, ex in enumerate(examples, 1):
            record = {
                "example_id": ex["example_id"],
                "model_suggestion": None,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "error": None,
            }
            try:
                record["model_suggestion"] = prelabel_one(client, ex["raw_text"])
                n_ok += 1
            except Exception as e:
                record["error"] = str(e)
                n_err += 1

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"[{i}/{len(examples)}] {ex['example_id']} -> "
                  f"{'ok' if record['error'] is None else 'ERROR: ' + record['error']}")

            time.sleep(0.2)  # light throttle, not a rate-limit-aware backoff

    print(f"\nDone. {n_ok} succeeded, {n_err} failed. Written to {out_path}")
    if n_err:
        print("Re-run failed example_ids manually if needed -- this script does not auto-retry.")


if __name__ == "__main__":
    main()
