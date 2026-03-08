"""
01_ingest_corpus.py — Parse, clean, and store the 20 Newsgroups corpus.

Usage:
    python scripts/01_ingest_corpus.py --tar data/20_newsgroups.tar.gz
    python scripts/01_ingest_corpus.py --dir data/20_newsgroups

Expected output:
    ~15,000-16,000 documents in data/corpus.db
    Cleaning report printed to console + saved as data/cleaning_report.json
"""

import os
import sys
import json
import tarfile
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import init_db, insert_document, get_doc_count
from app.preprocessor import clean_text, should_keep, compute_hash, CleaningStats


def parse_newsgroup_file(filepath: str):
    """
    Parse a single newsgroup file into (subject, raw_text).
    Files are structured as: headers, blank line, body.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            raw = f.read()
    except Exception:
        return None, None

    subject = ""
    for line in raw.split('\n')[:30]:
        if line.lower().startswith('subject:'):
            subject = line[8:].strip()
            break

    return subject, raw


def ingest_from_directory(corpus_dir: str, stats: CleaningStats):
    """Walk corpus directory, process each file."""
    seen_hashes = set()

    for newsgroup in sorted(os.listdir(corpus_dir)):
        ng_dir = os.path.join(corpus_dir, newsgroup)
        if not os.path.isdir(ng_dir):
            continue

        ng_count = 0
        for fname in sorted(os.listdir(ng_dir)):
            fpath = os.path.join(ng_dir, fname)
            if not os.path.isfile(fpath):
                continue

            stats.total_scanned += 1
            subject, raw = parse_newsgroup_file(fpath)
            if raw is None:
                stats.dropped_empty += 1
                continue

            stats.total_raw_tokens += len(raw.split())

            cleaned, raw_tokens, clean_tokens = clean_text(raw)
            content_hash = compute_hash(cleaned)
            keep, reason = should_keep(cleaned, clean_tokens, content_hash, seen_hashes)

            if not keep:
                if reason == "too_short":     stats.dropped_short += 1
                elif reason == "non_english": stats.dropped_non_english += 1
                elif reason == "duplicate":   stats.dropped_duplicate += 1
                elif reason == "empty":       stats.dropped_empty += 1
                continue

            seen_hashes.add(content_hash)
            doc_id = insert_document(newsgroup, subject, cleaned,
                                     clean_tokens, content_hash)
            if doc_id is not None:
                stats.kept += 1
                stats.total_clean_tokens += clean_tokens
                ng_count += 1

                if stats.kept % 1000 == 0:
                    elapsed = time.time() - _t_start
                    print(f"  [Ingest] {stats.kept} documents inserted... ({elapsed:.0f}s)")

        stats.per_newsgroup[newsgroup] = ng_count

    return stats


_t_start = time.time()


def main():
    global _t_start
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar", help="Path to 20_newsgroups.tar.gz")
    parser.add_argument("--dir", help="Path to extracted 20_newsgroups directory")
    args = parser.parse_args()

    init_db()

    corpus_dir = None

    if args.tar:
        print(f"[Ingest] Extracting {args.tar} → data/")
        with tarfile.open(args.tar, 'r:gz') as tf:
            tf.extractall("data/")
        # Find the extracted directory
        for name in os.listdir("data/"):
            if "newsgroup" in name.lower() and os.path.isdir(f"data/{name}"):
                corpus_dir = f"data/{name}"
                break
        if corpus_dir is None:
            print("ERROR: Could not find extracted directory in data/")
            sys.exit(1)
        print(f"[Ingest] Corpus dir: {corpus_dir}")
    elif args.dir:
        corpus_dir = args.dir
    else:
        # Auto-detect
        for candidate in ["data/20_newsgroups", "data/20news-18828"]:
            if os.path.isdir(candidate):
                corpus_dir = candidate
                break
        if corpus_dir is None:
            print("ERROR: Provide --tar or --dir argument.")
            print("  Example: python scripts/01_ingest_corpus.py --tar data/20_newsgroups.tar.gz")
            sys.exit(1)

    print(f"[Ingest] Processing corpus from: {corpus_dir}")
    print("[Ingest] Cleaning stages: headers | PGP | sigs | quotes | URLs | dedup | length")

    _t_start = time.time()
    stats = CleaningStats()
    ingest_from_directory(corpus_dir, stats)

    elapsed = time.time() - _t_start
    stats.print_report()

    # Save JSON report
    report = {
        "total_scanned": stats.total_scanned,
        "kept": stats.kept,
        "dropped_short": stats.dropped_short,
        "dropped_non_english": stats.dropped_non_english,
        "dropped_duplicate": stats.dropped_duplicate,
        "dropped_empty": stats.dropped_empty,
        "noise_pct": round(stats.noise_pct(), 3),
        "per_newsgroup": stats.per_newsgroup,
    }
    with open("data/cleaning_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[Ingest] Complete: {stats.kept} documents inserted. Time: {elapsed:.1f}s")
    print(f"[Ingest] Total documents in DB: {get_doc_count()}")
    print("[Ingest] Cleaning report saved to: data/cleaning_report.json")
    print("\nNext: python scripts/02_build_index.py")


if __name__ == "__main__":
    main()
