#!/usr/bin/env python3
"""
prepare_dataset.py

Memory-conscious dataset preparation pipeline for the TWCS (Customer Support
on Twitter) dataset, scoped to a single target brand (default: AppleSupport).

Two-pass design (bounded memory relative to a naive single full-text load):

  Pass 1 (chunked): stream the CSV once, building a compact per-tweet index
    of {author_id, in_response_to_tweet_id, created_at, inbound, quality
    flags} WITHOUT retaining tweet text. This index is used to (a) resolve
    each tweet's conversation root by walking in_response_to_tweet_id
    pointers upward, and (b) determine which roots the target brand
    participates in at all. Only tweet_id, a few short fields, and boolean
    flags are held in memory per row -- the large `text` column is not
    retained in this pass.

  Pass 2 (chunked): stream the CSV a second time, keeping only rows whose
    resolved root touches the target brand. Full row data (including text)
    is retained ONLY for this filtered subset, which is materially smaller
    than the full dataset.

IMPORTANT ASSUMPTION: `in_response_to_tweet_id` is single-valued and is the
correct field to walk for root reconstruction. `response_tweet_id` can be a
comma-separated list (a tweet can have multiple direct replies -- a branch
point) and is NOT used for root-walking; it is only used to flag branch
points as a reconstruction-anomaly signal, and stored in per-message output
for downstream use.

No API calls, no LLM calls. Deterministic: all sorts have an explicit,
stable tie-breaker (tweet_id) so re-running produces identical output.
"""
import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prepare_dataset")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Prepare a brand-scoped, thread-reconstructed corpus from TWCS.")
    p.add_argument("--input", default="data/twcs.csv", help="Path to raw twcs.csv")
    p.add_argument("--output-dir", default="data/processed", help="Directory for output files")
    p.add_argument("--brand", default="AppleSupport", help="Target brand author_id to scope the corpus to")
    p.add_argument("--chunksize", type=int, default=200_000, help="Rows per CSV chunk")
    return p.parse_args()


# ---------------------------------------------------------------------
# Timestamp parsing (TWCS format: "Tue Oct 31 22:10:47 +0000 2017")
# ---------------------------------------------------------------------
def parse_ts(raw):
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return datetime.strptime(raw, "%a %b %d %H:%M:%S %z %Y")
    except ValueError:
        return None


def norm_id(x):
    """Normalize an id field (tweet_id / in_response_to_tweet_id) to a clean
    string, treating NaN/empty/'nan' as missing (None)."""
    if x is None:
        return None
    if isinstance(x, float) and pd.isna(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    # guard against float-formatted ids like "123.0" sneaking in
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


# ---------------------------------------------------------------------
# Pass 1: compact index + root resolution
# ---------------------------------------------------------------------
def build_index_and_resolve_roots(input_path, chunksize, brand):
    """Returns:
        root_of: dict[tweet_id] -> root_id (str)
        meta: dict[tweet_id] -> dict(author_id, inbound, created_at, ts_valid)
        stats: dict of pass-1 level counters
    """
    parent_of = {}      # tweet_id -> in_response_to_tweet_id (or None)
    author_of = {}      # tweet_id -> author_id
    inbound_of = {}      # tweet_id -> bool
    ts_valid_of = {}     # tweet_id -> bool (created_at parsed successfully)
    seen_ids = set()
    dup_count = 0
    empty_text_count = 0
    malformed_ts_count = 0
    total_rows = 0

    dtype = {
        "tweet_id": str,
        "author_id": str,
        "in_response_to_tweet_id": str,
        "response_tweet_id": str,
        "text": str,
    }

    log.info("Pass 1: streaming %s in chunks of %d rows (building compact id index)...", input_path, chunksize)
    reader = pd.read_csv(input_path, dtype=dtype, chunksize=chunksize, keep_default_na=True)
    for chunk_idx, chunk in enumerate(reader):
        for row in chunk.itertuples(index=False):
            total_rows += 1
            tid = norm_id(row.tweet_id)
            if tid is None:
                continue
            if tid in seen_ids:
                dup_count += 1
                continue
            seen_ids.add(tid)

            author_id = row.author_id
            inbound = str(row.inbound).strip().lower() == "true"
            parent_id = norm_id(row.in_response_to_tweet_id)
            text = row.text
            text_empty = (not isinstance(text, str)) or (text.strip() == "")
            ts = parse_ts(row.created_at)
            ts_ok = ts is not None

            if text_empty:
                empty_text_count += 1
            if not ts_ok:
                malformed_ts_count += 1

            parent_of[tid] = parent_id
            author_of[tid] = author_id
            inbound_of[tid] = inbound
            ts_valid_of[tid] = ts_ok

        log.info("  Pass 1: processed chunk %d (%d rows so far)", chunk_idx + 1, total_rows)

    log.info("Pass 1 complete. total_rows=%d unique_ids=%d duplicates=%d empty_text=%d malformed_ts=%d",
              total_rows, len(seen_ids), dup_count, empty_text_count, malformed_ts_count)

    # Resolve root for every tweet_id via iterative parent-walk with memoization.
    log.info("Resolving conversation roots (walking in_response_to_tweet_id upward)...")
    root_of = {}
    cycle_guard_hits = 0

    def find_root(tid):
        nonlocal cycle_guard_hits
        path = []
        cur = tid
        steps = 0
        while True:
            if cur in root_of:
                root = root_of[cur]
                break
            parent = parent_of.get(cur)
            if parent is None or parent not in author_of:
                # parent missing from dataset entirely (e.g. truncated/foreign
                # thread) -- treat current node as its own root.
                root = cur
                break
            path.append(cur)
            if parent == cur or steps > 200:
                cycle_guard_hits += 1
                root = cur
                break
            cur = parent
            steps += 1
        for p in path:
            root_of[p] = root
        root_of[tid] = root
        return root

    all_ids = list(parent_of.keys())
    for i, tid in enumerate(all_ids):
        find_root(tid)
        if (i + 1) % 500_000 == 0:
            log.info("  Root resolution: %d / %d", i + 1, len(all_ids))

    log.info("Root resolution complete. %d tweets resolved to roots. cycle_guard_hits=%d",
              len(root_of), cycle_guard_hits)

    # Which roots does the target brand touch at all?
    brand_roots = set()
    for tid, author_id in author_of.items():
        if author_id == brand:
            brand_roots.add(root_of[tid])
    log.info("Roots touching brand '%s': %d", brand, len(brand_roots))

    stats = {
        "input_row_count": total_rows,
        "unique_tweet_ids": len(seen_ids),
        "duplicate_tweet_id_count": dup_count,
        "empty_text_count": empty_text_count,
        "malformed_timestamp_count": malformed_ts_count,
        "reconstructed_thread_count_all_brands": len(set(root_of.values())),
        "brand_touching_root_count": len(brand_roots),
        "cycle_guard_hit_count": cycle_guard_hits,
    }
    return root_of, author_of, inbound_of, ts_valid_of, brand_roots, stats


# ---------------------------------------------------------------------
# Pass 2: re-stream, keep only brand-relevant rows with full data
# ---------------------------------------------------------------------
def collect_brand_rows(input_path, chunksize, root_of, brand_roots, seen_ids_pass1):
    dtype = {
        "tweet_id": str,
        "author_id": str,
        "in_response_to_tweet_id": str,
        "response_tweet_id": str,
        "text": str,
    }
    kept_rows = []
    seen_again = set()
    log.info("Pass 2: streaming %s again, retaining only brand-relevant rows...", input_path)
    reader = pd.read_csv(input_path, dtype=dtype, chunksize=chunksize, keep_default_na=True)
    n_seen = 0
    for chunk_idx, chunk in enumerate(reader):
        for row in chunk.itertuples(index=False):
            n_seen += 1
            tid = norm_id(row.tweet_id)
            if tid is None or tid in seen_again:
                continue  # skip malformed/duplicate ids, already counted in pass 1
            root = root_of.get(tid)
            if root is None or root not in brand_roots:
                continue
            seen_again.add(tid)

            ts = parse_ts(row.created_at)
            kept_rows.append({
                "tweet_id": tid,
                "author_id": row.author_id,
                "inbound": str(row.inbound).strip().lower() == "true",
                "created_at_raw": row.created_at,
                "created_at": ts.isoformat() if ts else None,
                "timestamp_valid": ts is not None,
                "text": row.text if isinstance(row.text, str) else "",
                "text_empty": (not isinstance(row.text, str)) or (row.text.strip() == ""),
                "root_id": root,
                "in_response_to_tweet_id": norm_id(row.in_response_to_tweet_id),
                "response_tweet_id_raw": row.response_tweet_id if isinstance(row.response_tweet_id, str) else None,
                "is_branch_point": isinstance(row.response_tweet_id, str) and "," in row.response_tweet_id,
            })
        log.info("  Pass 2: scanned chunk %d (%d total rows scanned, %d kept)", chunk_idx + 1, n_seen, len(kept_rows))

    log.info("Pass 2 complete. Kept %d brand-relevant rows out of %d scanned.", len(kept_rows), n_seen)
    return kept_rows


# ---------------------------------------------------------------------
# Thread assembly: group kept rows by root, sort by created_at (tie-break
# tweet_id for determinism), assign turn index, determine coherence.
# ---------------------------------------------------------------------
def assemble_threads(kept_rows, brand):
    by_root = defaultdict(list)
    for r in kept_rows:
        by_root[r["root_id"]].append(r)

    threads = []
    messages = []
    length_hist = defaultdict(int)
    coherent_count = 0
    multi_customer_flagged = 0
    unparseable_ts_in_thread = 0

    for root_id, rows in by_root.items():
        # deterministic sort: valid timestamps first (chronological), then
        # any unparseable-timestamp rows appended in tweet_id order so the
        # thread is still fully deterministic even with a bad timestamp.
        def sort_key(r):
            return (0, r["created_at"], int(r["tweet_id"])) if r["timestamp_valid"] else (1, "", int(r["tweet_id"]))

        rows_sorted = sorted(rows, key=sort_key)
        if any(not r["timestamp_valid"] for r in rows_sorted):
            unparseable_ts_in_thread += 1

        inbound_authors = {r["author_id"] for r in rows_sorted if r["inbound"]}
        is_coherent = len(inbound_authors) <= 1  # 0 (brand-only thread) or 1 distinct customer
        quality_flag = None if is_coherent else "multi_customer_root"
        if is_coherent:
            coherent_count += 1
        else:
            multi_customer_flagged += 1

        turns = []
        for i, r in enumerate(rows_sorted):
            turn = dict(r)
            turn["turn_index"] = i
            turn["thread_id"] = root_id
            turn["role"] = "customer" if r["inbound"] else "brand"
            turns.append(turn)
            messages.append(turn)

        length_hist[len(turns)] += 1

        threads.append({
            "thread_id": root_id,
            "root_id": root_id,
            "is_coherent": is_coherent,
            "quality_flag": quality_flag,
            "n_turns": len(turns),
            "n_distinct_customers": len(inbound_authors),
            "has_customer_message": any(t["inbound"] for t in turns),
            "has_brand_message": any((not t["inbound"]) for t in turns),
            "turns": turns,
        })

    stats = {
        "reconstructed_thread_count": len(threads),
        "valid_coherent_thread_count": coherent_count,
        "multi_customer_flagged_thread_count": multi_customer_flagged,
        "threads_with_unparseable_timestamp": unparseable_ts_in_thread,
        "thread_length_distribution": dict(sorted(length_hist.items())),
    }
    return threads, messages, stats


def main():
    args = parse_args()

    root_of, author_of, inbound_of, ts_valid_of, brand_roots, pass1_stats = build_index_and_resolve_roots(
        args.input, args.chunksize, args.brand
    )
    kept_rows = collect_brand_rows(args.input, args.chunksize, root_of, brand_roots, seen_ids_pass1=None)
    threads, messages, assemble_stats = assemble_threads(kept_rows, args.brand)

    customer_msg_count = sum(1 for m in messages if m["inbound"])
    brand_msg_count = sum(1 for m in messages if not m["inbound"])
    valid_ts = [m["created_at"] for m in messages if m["timestamp_valid"]]
    date_range = {"min": min(valid_ts) if valid_ts else None, "max": max(valid_ts) if valid_ts else None}
    branch_point_count = sum(1 for m in messages if m["is_branch_point"])

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    threads_path = os.path.join(args.output_dir, f"{args.brand.lower()}_threads.jsonl")
    messages_path = os.path.join(args.output_dir, f"{args.brand.lower()}_messages.jsonl")
    stats_path = os.path.join(args.output_dir, "dataset_stats.json")

    log.info("Writing %s ...", threads_path)
    with open(threads_path, "w", encoding="utf-8") as f:
        for t in threads:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    log.info("Writing %s ...", messages_path)
    with open(messages_path, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    stats = {
        "brand": args.brand,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **pass1_stats,
        **assemble_stats,
        "brand_customer_message_count": customer_msg_count,
        "brand_reply_message_count": brand_msg_count,
        "branch_point_message_count": branch_point_count,
        "date_range": date_range,
    }
    log.info("Writing %s ...", stats_path)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    log.info("Done.")
    log.info("Summary: %s", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
