#!/usr/bin/env python3
"""
build_evidence_corpus.py

Deterministic construction of retrieval evidence units from the processed,
brand-scoped AppleSupport thread corpus. No API calls, no LLM calls. Every
classification decision comes from the calibrated rule-based
`reply_classification.py` (95.4% agreement against a 65-thread manual audit;
see reply_classification.py's module docstring and the calibration report
for the 3 known, named residual disagreements -- not hidden here).

One evidence unit is produced per eligible thread:
  - customer_text: the maximal consecutive prefix of customer turns
    starting at turn_index=0, before the first brand turn.
  - brand_evidence_text: ONLY brand turns classified `substantive`,
    concatenated in chronological order. Diagnostic-only, redirect,
    acknowledgement, and unclassified turns are excluded from this text but
    preserved in `excluded_brand_tweet_ids` / `brand_reply_classifications`
    for full traceability.

Eligibility (a thread produces an evidence unit only if ALL hold):
  - is_coherent == true and quality_flag is null (not multi_customer_root)
  - has both a customer message and a brand message
  - root_id is NOT in the excluded-root-ids file (golden/dev leakage guard)
  - turn_index=0 is a customer turn (guards against a brand-initiated
    anomaly, which would otherwise produce an empty customer_text)
  - at least one brand turn classifies as `substantive`

Determinism:
  - no wall-clock value is embedded in any evidence unit
  - output is sorted by root_id (numeric) regardless of input file order,
    so reruns are stable even if upstream iteration order ever changes
  - all classification is pure-function rule-based (reply_classification.py)
"""
import argparse
import json
import sys
from collections import Counter

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from reply_classification import classify_brand_reply


def parse_args():
    p = argparse.ArgumentParser(description="Build deterministic retrieval evidence units from processed AppleSupport threads.")
    p.add_argument("--input", default="data/processed/applesupport_threads.jsonl")
    p.add_argument("--output", default="data/processed/evidence_corpus.jsonl")
    p.add_argument("--excluded-root-ids-file", default=None,
                    help="Newline-separated root_ids to exclude (golden/dev leakage guard).")
    return p.parse_args()


def load_excluded_roots(path):
    if not path:
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def opening_customer_prefix(turns):
    """Maximal consecutive prefix of customer turns starting at index 0."""
    prefix = []
    for t in turns:
        if t["role"] == "customer":
            prefix.append(t)
        else:
            break
    return prefix


def build_evidence_unit(thread, source_file):
    turns = thread["turns"]
    if not turns or turns[0]["role"] != "customer":
        return None, "brand_initiated_anomaly"

    customer_turns = opening_customer_prefix(turns)
    if not customer_turns:
        return None, "brand_initiated_anomaly"

    brand_turns = [t for t in turns if t["role"] == "brand"]
    if not brand_turns:
        return None, "no_brand_message"  # should already be filtered upstream; defensive

    classifications = [(t["tweet_id"], classify_brand_reply(t["text"])) for t in brand_turns]

    substantive_ids, substantive_texts = [], []
    excluded_ids = []
    for (tid, label), t in zip(classifications, brand_turns):
        if label == "substantive":
            substantive_ids.append(tid)
            substantive_texts.append(t["text"])
        else:
            excluded_ids.append(tid)

    if not substantive_ids:
        return None, "no_substantive_brand_evidence"

    customer_text = "\n".join(t["text"] for t in customer_turns)
    brand_evidence_text = "\n".join(substantive_texts)

    if not customer_text.strip():
        return None, "empty_customer_text"
    if not brand_evidence_text.strip():
        return None, "empty_brand_evidence_text"

    evidence = {
        "evidence_id": f"ev_{thread['root_id']}",
        "thread_id": thread["thread_id"],
        "root_id": thread["root_id"],
        "customer_tweet_ids": [t["tweet_id"] for t in customer_turns],
        "brand_tweet_ids": substantive_ids,
        "excluded_brand_tweet_ids": excluded_ids,
        "customer_text": customer_text,
        "brand_evidence_text": brand_evidence_text,
        "created_at": customer_turns[0].get("created_at"),
        "thread_quality": {
            "is_coherent": thread["is_coherent"],
            "quality_flag": thread["quality_flag"],
            "n_turns": thread["n_turns"],
            "n_distinct_customers": thread["n_distinct_customers"],
        },
        "evidence_type": "visible_resolution",
        "brand_reply_classifications": [
            {"tweet_id": tid, "label": label} for tid, label in classifications
        ],
        "n_substantive_brand_replies": len(substantive_ids),
        "n_total_brand_replies_in_thread": len(brand_turns),
        "source_files": [source_file],
    }
    return evidence, None


def main():
    args = parse_args()
    excluded_roots = load_excluded_roots(args.excluded_root_ids_file)

    counters = Counter()
    excluded_brand_reply_counts = Counter()
    evidence_units = []
    anomalies = []

    with open(args.input) as f:
        for line in f:
            thread = json.loads(line)
            counters["total_input_threads"] += 1

            if not thread["is_coherent"] or thread["quality_flag"] is not None:
                counters["incoherent_or_multi_customer_excluded"] += 1
                continue
            counters["coherent_threads"] += 1

            if not thread["has_customer_message"]:
                counters["brand_only_excluded"] += 1
                continue
            if not thread["has_brand_message"]:
                counters["customer_only_unanswered_excluded"] += 1
                continue

            if thread["root_id"] in excluded_roots:
                counters["excluded_golden_dev_roots"] += 1
                continue

            evidence, reason = build_evidence_unit(thread, args.input)
            if evidence is None:
                if reason == "no_substantive_brand_evidence":
                    counters["no_substantive_brand_evidence"] += 1
                elif reason == "brand_initiated_anomaly":
                    counters["brand_initiated_anomaly"] += 1
                    anomalies.append({"root_id": thread["root_id"], "reason": reason})
                elif reason == "empty_customer_text":
                    counters["empty_customer_text_excluded"] += 1
                    anomalies.append({"root_id": thread["root_id"], "reason": reason})
                elif reason == "empty_brand_evidence_text":
                    counters["empty_brand_evidence_text_excluded"] += 1
                    anomalies.append({"root_id": thread["root_id"], "reason": reason})
                else:
                    counters["no_brand_message_defensive"] += 1
                continue

            for item in evidence["brand_reply_classifications"]:
                if item["label"] != "substantive":
                    excluded_brand_reply_counts[item["label"]] += 1
            counters["total_substantive_replies"] += evidence["n_substantive_brand_replies"]
            evidence_units.append(evidence)

    counters["eligible_threads"] = len(evidence_units)
    counters["evidence_units_produced"] = len(evidence_units)
    counters["evidence_units_1_substantive"] = sum(1 for e in evidence_units if e["n_substantive_brand_replies"] == 1)
    counters["evidence_units_gt1_substantive"] = sum(1 for e in evidence_units if e["n_substantive_brand_replies"] > 1)

    # deterministic output order: sort by numeric root_id regardless of input order
    evidence_units.sort(key=lambda e: int(e["root_id"]))

    # integrity checks (all must be zero)
    evidence_ids = [e["evidence_id"] for e in evidence_units]
    root_ids = [e["root_id"] for e in evidence_units]
    dup_evidence_id = len(evidence_ids) - len(set(evidence_ids))
    dup_root_id = len(root_ids) - len(set(root_ids))
    from_excluded = sum(1 for e in evidence_units if e["root_id"] in excluded_roots)
    from_incoherent = sum(1 for e in evidence_units if not e["thread_quality"]["is_coherent"] or e["thread_quality"]["quality_flag"] is not None)
    empty_customer = sum(1 for e in evidence_units if not e["customer_text"].strip())
    empty_brand = sum(1 for e in evidence_units if not e["brand_evidence_text"].strip())

    integrity = {
        "duplicate_evidence_id_count": dup_evidence_id,
        "duplicate_root_id_count": dup_root_id,
        "evidence_units_from_excluded_roots": from_excluded,
        "evidence_units_from_incoherent_threads": from_incoherent,
        "evidence_units_with_empty_customer_text": empty_customer,
        "evidence_units_with_empty_brand_evidence_text": empty_brand,
    }
    for k, v in integrity.items():
        assert v == 0, f"INTEGRITY CHECK FAILED: {k} = {v}, expected 0"

    with open(args.output, "w", encoding="utf-8") as f:
        for e in evidence_units:
            f.write(json.dumps(e, ensure_ascii=False, sort_keys=False) + "\n")

    report = {
        "counters": dict(counters),
        "excluded_brand_replies_by_classification": dict(excluded_brand_reply_counts),
        "integrity_checks_all_must_be_zero": integrity,
        "anomalies_sample": anomalies[:20],
        "anomaly_total_count": len(anomalies),
        "classifier_calibration_note": (
            "reply_classification.py rules are calibrated against a 65-thread "
            "manual audit: 62/65 (95.4%) agreement. 3 known residual "
            "disagreements are documented in the calibration report and in "
            "reply_classification.py's module docstring, not hidden here."
        ),
    }
    report_path = args.output.replace(".jsonl", "_stats.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(evidence_units)} evidence units to {args.output}")
    print(f"Wrote validation report to {report_path}")


if __name__ == "__main__":
    main()
