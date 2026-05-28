#!/usr/bin/env python3
"""
LLM-assisted weak labeling for Reddit climate comments.

Uses taxonomy definitions + few-shot manual examples. Outputs silver labels only.

Usage:
  export OPENAI_API_KEY=sk-...
  python src/llm_label_comments.py --max-rows 100 --priority-queue-first
  python src/llm_label_comments.py --dry-run --max-rows 20
  python src/llm_label_comments.py --batch-size 5 --checkpoint-every 25

Requires: pip install openai
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from label_utils import (
    AUTO_LABELED,
    OUTPUTS,
    PROCESSED,
    body_hash,
    load_manual_labeled,
)
from taxonomy import CANONICAL_LABELS, TAXONOMY, normalize_label

ROOT = Path(__file__).resolve().parents[1]
REDDIT_UNLABELED = PROCESSED / "preprocessed_comments_400000.csv"
LLM_OUTPUT = OUTPUTS / "llm_labeled_comments.csv"
LLM_CHECKPOINT = OUTPUTS / "llm_label_checkpoint.jsonl"

# User-requested priority (includes nihilism critique)
LLM_PRIORITY_LABELS = frozenset(
    {
        "Climate nihilism",
        "climate anxiety",
        "Climate nihilism critique",
        "Climate denial",  # also important for project
    }
)

CONFUSED_LABELS = ["climate anxiety", "Climate nihilism", "Climate nihilism critique"]

MAX_BODY_CHARS = 3500
MAX_EXAMPLE_CHARS = 600


@dataclass
class LabelResult:
    body: str
    body_hash: str
    predicted_label: str
    reasoning: str
    confidence: float
    few_shot_examples_used: str
    model: str
    error: str = ""


def build_taxonomy_block() -> str:
    lines = ["# Climate opinion taxonomy (choose exactly ONE label)\n"]
    for label in CANONICAL_LABELS:
        lines.append(f"## {label}\n{TAXONOMY[label]}\n")
    lines.append(
        """
## Critical distinctions (read carefully)

### climate anxiety vs Climate nihilism
- **climate anxiety**: Fear, worry, dread about climate — the author is distressed but does NOT say action is pointless.
- **Climate nihilism**: Hopelessness, futility, "too late", "nothing we can do", inevitable collapse — giving up on stopping climate change.

### Climate nihilism vs Climate nihilism critique
- **Climate nihilism**: The AUTHOR expresses doomism/futility themselves.
- **Climate nihilism critique**: The author criticizes or pushes back against doomist/nihilistic views (e.g. "stop being hopeless", "we still have time").

### Climate denial vs Climate denial critique
- **Climate denial**: Author denies climate change is real/serious.
- **Climate denial critique**: Author criticizes deniers.
"""
    )
    return "\n".join(lines)


def select_few_shot_examples(
    manual: pd.DataFrame,
    per_class: int = 1,
    extra_classes: Optional[Sequence[str]] = None,
    extra_per_class: int = 2,
) -> Tuple[List[dict], str]:
    """Return example dicts and a compact description for the output column."""
    extra_classes = list(extra_classes or CONFUSED_LABELS)
    examples: List[dict] = []
    used_labels: List[str] = []

    for label in CANONICAL_LABELS:
        n = extra_per_class if label in extra_classes else per_class
        subset = manual[manual["label"] == label]
        if subset.empty:
            continue
        sample = subset.sample(n=min(n, len(subset)), random_state=42)
        for _, row in sample.iterrows():
            body = str(row["body"])[:MAX_EXAMPLE_CHARS]
            examples.append({"label": label, "text": body})
            used_labels.append(label)

    desc = "; ".join(sorted(set(used_labels)))
    return examples, desc


def build_system_prompt(few_shot: List[dict]) -> str:
    taxonomy = build_taxonomy_block()
    few_shot_block = "\n".join(
        f'Example ({ex["label"]}):\n"""{ex["text"]}"""\n' for ex in few_shot
    )
    return f"""You are an expert annotator for climate-related Reddit comments.
Assign exactly ONE label from the taxonomy below. These are WEAK/SILVER labels for research — be careful on edge cases.

{taxonomy}

# Few-shot examples (gold standard style)
{few_shot_block}

# Output rules
- Respond with valid JSON only (no markdown).
- For each comment, output: id, predicted_label, confidence (0.0-1.0), reasoning (one sentence).
- predicted_label must match a taxonomy heading EXACTLY (case-sensitive).
- If the comment is not about climate opinions, use "Not climate opinion".
- Lower confidence when sarcasm, ambiguity, or mixed opinions appear.
"""


def build_batch_user_prompt(items: List[Tuple[str, str]]) -> str:
    """items: list of (id, body)"""
    blocks = []
    for cid, body in items:
        text = body[:MAX_BODY_CHARS].replace('"""', "'")
        blocks.append(f'Comment id="{cid}":\n"""{text}"""\n')
    return (
        "Label each comment below. Return JSON:\n"
        '{"results": [{"id": "...", "predicted_label": "...", '
        '"confidence": 0.0, "reasoning": "..."}]}\n\n'
        + "\n".join(blocks)
    )


def parse_llm_json(content: str) -> List[dict]:
    content = content.strip()
    # Strip markdown fences if present
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    data = json.loads(content)
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    if isinstance(data, list):
        return data
    raise ValueError("Unexpected JSON shape from LLM")


def call_openai_batch(
    client: Any,
    model: str,
    system_prompt: str,
    items: List[Tuple[str, str]],
    temperature: float = 0.0,
    max_retries: int = 3,
) -> List[dict]:
    user_prompt = build_batch_user_prompt(items)
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = resp.choices[0].message.content or "{}"
            return parse_llm_json(content)
        except Exception as e:
            last_err = e
            time.sleep(2**attempt)
    raise RuntimeError(f"LLM call failed after {max_retries} tries: {last_err}")


def dry_run_batch(
    items: List[Tuple[str, str]],
    few_shot_desc: str,
) -> List[dict]:
    """Offline placeholder when no API key — uses keyword heuristics."""
    from auto_label_comments import compile_keyword_patterns, keyword_scores

    compiled = compile_keyword_patterns()
    out = []
    for cid, body in items:
        scores, _ = keyword_scores(body, compiled)
        label = max(scores, key=scores.get)
        top = scores[label]
        total = sum(scores.values()) or 1
        conf = min(0.85, 0.35 + top / total * 0.5)
        out.append(
            {
                "id": cid,
                "predicted_label": label,
                "confidence": round(conf, 3),
                "reasoning": "dry-run: keyword heuristic only (set OPENAI_API_KEY for real LLM)",
            }
        )
    return out


def load_completed_hashes() -> Set[str]:
    hashes: Set[str] = set()
    for path in (LLM_CHECKPOINT, LLM_OUTPUT):
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    hashes.add(json.loads(line)["body_hash"])
                except (json.JSONDecodeError, KeyError):
                    pass
        else:
            try:
                df = pd.read_csv(path, usecols=["body_hash"], encoding="utf-8")
                hashes.update(df["body_hash"].astype(str))
            except Exception:
                pass
    return hashes


def append_checkpoint(rows: List[LabelResult]) -> None:
    LLM_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with LLM_CHECKPOINT.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {
                        "body_hash": r.body_hash,
                        "body": r.body,
                        "predicted_label": r.predicted_label,
                        "reasoning": r.reasoning,
                        "confidence": r.confidence,
                        "few_shot_examples_used": r.few_shot_examples_used,
                        "model": r.model,
                        "error": r.error,
                        "ts": datetime.now(timezone.utc).isoformat(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def flush_output_csv(all_rows: List[LabelResult]) -> None:
    LLM_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "body_hash": r.body_hash,
                "body": r.body,
                "predicted_label": r.predicted_label,
                "reasoning": r.reasoning,
                "confidence": r.confidence,
                "few_shot_examples_used": r.few_shot_examples_used,
                "model": r.model,
                "error": r.error,
            }
            for r in all_rows
        ]
    )
    df.to_csv(LLM_OUTPUT, index=False, quoting=1)


def load_unlabeled_queue(
    source: str,
    max_rows: Optional[int],
    priority_queue_first: bool,
    exclude_hashes: Set[str],
) -> pd.DataFrame:
    if source == "reddit":
        df = pd.read_csv(
            REDDIT_UNLABELED,
            usecols=["body"],
            encoding="utf-8",
            on_bad_lines="skip",
            nrows=max_rows,
        )
        df["body"] = df["body"].fillna("").astype(str)
        df["predicted_label"] = ""
        df["confidence"] = 0.0
        df["needs_human_review"] = True
    elif source == "auto":
        usecols = ["body", "predicted_label", "confidence", "needs_human_review"]
        df = pd.read_csv(
            AUTO_LABELED,
            usecols=usecols,
            encoding="utf-8",
            low_memory=False,
        )
        df["body"] = df["body"].fillna("").astype(str)
        df["predicted_label"] = df["predicted_label"].map(normalize_label)
    else:
        raise ValueError(f"Unknown source: {source}")

    df = df[df["body"].str.len() > 10].copy()
    df["body_hash"] = df["body"].map(body_hash)
    df = df[~df["body_hash"].isin(exclude_hashes)]

    if priority_queue_first:
        is_priority_pred = df["predicted_label"].isin(LLM_PRIORITY_LABELS)
        is_nih_crit = df["predicted_label"] == "Climate nihilism critique"
        df["_prio"] = 0
        df.loc[is_priority_pred, "_prio"] = 2
        df.loc[is_nih_crit, "_prio"] = 3
        # Boost rows weak-labeler thought were nihilism/anxiety but uncertain
        if "needs_human_review" in df.columns:
            review = df["needs_human_review"].astype(str).str.lower().isin(
                ["true", "1", "yes"]
            )
            df.loc[review & is_priority_pred, "_prio"] = 4
        df = df.sort_values(
            ["_prio", "confidence"],
            ascending=[False, True],
        ).drop(columns=["_prio"], errors="ignore")
    else:
        df = df.sample(frac=1, random_state=42)

    if max_rows:
        df = df.head(max_rows)
    return df.reset_index(drop=True)


def label_batch(
    client: Optional[Any],
    model: str,
    system_prompt: str,
    few_shot_desc: str,
    batch_df: pd.DataFrame,
    dry_run: bool,
) -> List[LabelResult]:
    items = [(str(row.body_hash), str(row.body)) for row in batch_df.itertuples()]
    if dry_run:
        parsed = dry_run_batch(items, few_shot_desc)
    else:
        parsed = call_openai_batch(client, model, system_prompt, items)

    by_id = {str(p.get("id", "")): p for p in parsed}
    results: List[LabelResult] = []
    for row in batch_df.itertuples():
        cid = str(row.body_hash)
        p = by_id.get(cid, {})
        raw_label = p.get("predicted_label", "")
        label = normalize_label(raw_label) or "Not climate opinion"
        if label not in CANONICAL_LABELS:
            label = "climate opinion critique"
        try:
            conf = float(p.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        conf = max(0.0, min(1.0, conf))
        results.append(
            LabelResult(
                body=str(row.body),
                body_hash=cid,
                predicted_label=label,
                reasoning=str(p.get("reasoning", ""))[:500],
                confidence=conf,
                few_shot_examples_used=few_shot_desc,
                model="dry-run" if dry_run else model,
                error="" if cid in by_id else "missing_from_llm_response",
            )
        )
    return results


def run_pipeline(args: argparse.Namespace) -> None:
    manual = load_manual_labeled()
    few_shot, few_shot_desc = select_few_shot_examples(
        manual,
        per_class=args.few_shot_per_class,
        extra_per_class=args.few_shot_extra_confused,
    )
    system_prompt = build_system_prompt(few_shot)
    print(f"Few-shot pool: {len(few_shot)} examples covering: {few_shot_desc}")

    completed = load_completed_hashes()
    print(f"Skipping {len(completed)} already labeled (checkpoint/output).")

    queue = load_unlabeled_queue(
        source=args.source,
        max_rows=args.max_rows,
        priority_queue_first=args.priority_queue_first,
        exclude_hashes=completed,
    )
    print(f"Queue size: {len(queue)} comments")
    if queue.empty:
        print("Nothing to label.")
        return

    client = None
    dry_run = args.dry_run
    if not dry_run:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            print(
                "OPENAI_API_KEY not set — use --dry-run for offline test, or export your key.",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        except ImportError:
            print("Install openai: pip install openai", file=sys.stderr)
            sys.exit(1)

    all_results: List[LabelResult] = []
    # Reload prior results for final CSV merge
    if LLM_CHECKPOINT.exists():
        for line in LLM_CHECKPOINT.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            all_results.append(
                LabelResult(
                    body=o["body"],
                    body_hash=o["body_hash"],
                    predicted_label=o["predicted_label"],
                    reasoning=o.get("reasoning", ""),
                    confidence=float(o.get("confidence", 0)),
                    few_shot_examples_used=o.get("few_shot_examples_used", ""),
                    model=o.get("model", ""),
                    error=o.get("error", ""),
                )
            )

    batch_size = args.batch_size
    n = len(queue)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_df = queue.iloc[start:end]
        print(f"Labeling {start + 1}-{end} / {n}…")
        try:
            batch_results = label_batch(
                client,
                args.model,
                system_prompt,
                few_shot_desc,
                batch_df,
                dry_run=dry_run,
            )
        except Exception as e:
            print(f"  batch failed: {e}")
            batch_results = [
                LabelResult(
                    body=str(r.body),
                    body_hash=str(r.body_hash),
                    predicted_label="",
                    reasoning="",
                    confidence=0.0,
                    few_shot_examples_used=few_shot_desc,
                    model=args.model,
                    error=str(e),
                )
                for r in batch_df.itertuples()
            ]

        all_results.extend(batch_results)
        append_checkpoint(batch_results)

        if (end % args.checkpoint_every < batch_size) or end == n:
            flush_output_csv(all_results)
            print(f"  checkpoint -> {LLM_CHECKPOINT} ({end} rows)")
            print(f"  csv -> {LLM_OUTPUT}")

        if args.sleep_seconds > 0 and not dry_run:
            time.sleep(args.sleep_seconds)

    flush_output_csv(all_results)
    print(f"\nDone. Wrote {len(all_results)} rows -> {LLM_OUTPUT}")
    if all_results:
        vc = pd.Series([r.predicted_label for r in all_results]).value_counts()
        print(vc.head(12))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM weak-label Reddit climate comments.")
    p.add_argument(
        "--source",
        choices=["auto", "reddit"],
        default="auto",
        help="auto = auto_labeled_comments.csv (supports priority queue); reddit = raw 400k",
    )
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument(
        "--priority-queue-first",
        action="store_true",
        help="Label nihilism / anxiety / nihilism critique (and denial) candidates first",
    )
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--few-shot-per-class", type=int, default=1)
    p.add_argument("--few-shot-extra-confused", type=int, default=2)
    p.add_argument("--sleep-seconds", type=float, default=0.5)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="No API calls; keyword heuristic placeholder",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
