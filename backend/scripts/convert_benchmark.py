"""
convert_benchmark.py — turn a downloaded benchmark into a detector dataset.

Manual operator tool (like the rest of `backend/scripts/`, it is never
pytest-collected — see backend/pytest.ini). The conversion logic itself lives
in `eval/adapters.py` and is tested there; this is only the file plumbing.

It does not download anything. Which benchmarks to use is a licensing and
prioritization decision, and the corpora are large — fetch them yourself, then
point this at the file.

    # HaluEval (self-contained: each row carries knowledge + both answers)
    python scripts/convert_benchmark.py halueval \\
        --input qa_data.json --task qa --output ../data/benchmarks/halueval_qa.jsonl

    # FEVER (needs the wiki dump to resolve evidence pointers to sentences).
    # The dump is streamed into an on-disk SQLite index once (~5.4M sentences);
    # peak memory stays in the tens of MB rather than the 4-6 GB an in-memory
    # dict would cost. Later runs reuse the index and can omit --wiki-dir.
    python scripts/convert_benchmark.py fever \\
        --input train.jsonl --wiki-dir wiki-pages/ \\
        --output ../data/benchmarks/fever_train.jsonl

    # anything else
    python scripts/convert_benchmark.py generic --input rows.jsonl \\
        --claim-field claim --premise-field context --label-field label \\
        --output ../data/benchmarks/other.jsonl

Then score it with no API key and no retrieval:

    curl -X POST localhost:8000/api/eval/detector \\
         -H 'content-type: application/json' \\
         -d '{"dataset_path": "../data/benchmarks/halueval_qa.jsonl"}'
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.adapters import (  # noqa: E402
    from_fever, from_generic, from_halueval, write_detector_dataset,
)
from eval.fever_wiki import WikiSentenceIndex  # noqa: E402


def _load_rows(path: str) -> list[dict]:
    """Accept either JSONL or a JSON array — benchmarks ship as both."""
    with open(path, encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            return json.load(f)
        rows = []
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  ! skipping {path}:{lineno} — {exc}", file=sys.stderr)
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("format", choices=["halueval", "fever", "generic"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", default="qa", help="HaluEval: qa | dialogue | summarization")
    parser.add_argument("--wiki-dir", help="FEVER: directory of wiki-pages JSONL files")
    parser.add_argument(
        "--wiki-index", default="../data/benchmarks/fever_wiki.sqlite",
        help="FEVER: where to build/reuse the on-disk sentence index. Built once "
             "from --wiki-dir, then reused; peak memory stays in the tens of MB "
             "regardless of dump size.",
    )
    parser.add_argument("--rebuild-wiki-index", action="store_true",
                        help="FEVER: re-index even if the index already exists.")
    parser.add_argument(
        "--keep-unresolvable-nei", action="store_true",
        help="FEVER: keep NOT ENOUGH INFO claims that have no gold evidence. "
             "These get an EMPTY premise, so the empty-premise guard flags them "
             "automatically and they score as correct catches without the model "
             "ever running — inflating NEI recall toward 1.0. Off by default.",
    )
    parser.add_argument("--claim-field", default="claim")
    parser.add_argument("--premise-field", default="premise")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--id-field")
    args = parser.parse_args()

    rows = _load_rows(args.input)
    print(f"Loaded {len(rows):,} rows from {args.input}")

    if args.format == "halueval":
        items, report = from_halueval(rows, task=args.task)
    elif args.format == "fever":
        resolve, index = None, None
        have_index = os.path.exists(args.wiki_index) and not args.rebuild_wiki_index
        if args.wiki_dir or have_index:
            if args.wiki_dir:
                print(f"Building wiki index at {args.wiki_index} from {args.wiki_dir} ...")
                build = WikiSentenceIndex.build(
                    args.wiki_dir, args.wiki_index, force=args.rebuild_wiki_index,
                    progress=lambda name, n: print(f"  {name}: {n:,} sentences so far"),
                )
                print(f"  {build['status']}: {build['sentences']:,} sentences")
            index = WikiSentenceIndex(args.wiki_index)
            print(f"  index ready ({len(index):,} sentences)")
            resolve = index.as_resolver()
        else:
            print("  ! no --wiki-dir and no existing --wiki-index: evidence cannot be", file=sys.stderr)
            print("    resolved, so every SUPPORTS/REFUTES row will be dropped.", file=sys.stderr)
        try:
            items, report = from_fever(
                rows, resolve_evidence=resolve, keep_unresolvable_nei=args.keep_unresolvable_nei
            )
        finally:
            if index is not None:
                index.close()
    else:
        items, report = from_generic(
            rows, claim_field=args.claim_field, premise_field=args.premise_field,
            label_field=args.label_field, id_field=args.id_field,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    write_detector_dataset(items, args.output)

    print(f"\nWrote {len(items):,} detector items to {args.output}")
    print("Conversion report:", json.dumps(report, indent=2))
    counts = {}
    for it in items:
        counts[it["label"]] = counts.get(it["label"], 0) + 1
    print("Label mix:", counts)
    if len(counts) < 2:
        print("  ! only one label class present — precision/recall degenerate "
              "and AUROC is undefined.", file=sys.stderr)


if __name__ == "__main__":
    main()
