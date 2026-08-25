"""
step6_pipeline.py

End-to-end orchestration pipeline for the HTS RAG system.

Ties together all previous build steps (1–5) plus the optional knowledge
graph builder (step7_knowledgegraph.py) into a single command.

Stages (in order):
  collect  -> step1_collect.py    -> hts_full.json
  chunk    -> step2_chunk.py      -> hts_chunks.json
  faiss    -> step3_faisscpu.py   -> indexes/faiss_index.bin + chunks_metadata.json
  bm25     -> step4_bm25_index.py -> indexes/bm25_index.pkl  + bm25_chunk_ids.json
  graph    -> step7_knowledgegraph.py -> indexes/hts_knowledge_graph.graphml + .gpickle
  verify   -> smoke-tests HybridRetriever with 3 sample queries in-process

Usage
-----
  # Show status of all artifact files:
  python step6_pipeline.py status

  # Dry-run (print what would run, change nothing):
  python step6_pipeline.py build --dry-run

  # Full build (skips stages whose output files already exist):
  python step6_pipeline.py build

  # Force-rebuild everything from scratch:
  python step6_pipeline.py build --force

  # Skip specific stages (e.g. data you already have):
  python step6_pipeline.py build --skip collect chunk

  # Run only specific stages:
  python step6_pipeline.py build --only faiss bm25 verify

  # Interactive search (loads HybridRetriever and prompts):
  python step6_pipeline.py search
  python step6_pipeline.py search "purebred breeding horses"

  # Run a single step by number (1-6):
  python step6_pipeline.py run-step 3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Artifact / file declarations for each stage
# ---------------------------------------------------------------------------

# Project root is wherever this file lives.
_HERE = Path(__file__).resolve().parent

STAGE_META = {
    "collect": {
        "label": "Collect raw HTS data",
        "script": "step1_collect.py",
        "outputs": ["hts_full.json"],
        "description": "Fetches chapters 01-99 from the USITC REST API.",
    },
    "chunk": {
        "label": "Build hierarchy chunks",
        "script": "step2_chunk.py",
        "outputs": ["hts_chunks.json"],
        "description": "Reconstructs parent breadcrumbs and inherits duty rates.",
    },
    "faiss": {
        "label": "Build FAISS dense index",
        "script": "step3_faisscpu.py",
        "outputs": ["indexes/faiss_index.bin", "indexes/chunks_metadata.json"],
        "description": "Encodes chunks with all-MiniLM-L6-v2 and builds IndexFlatIP.",
    },
    "bm25": {
        "label": "Build BM25 sparse index",
        "script": "step4_bm25_index.py",
        "outputs": ["indexes/bm25_index.pkl", "indexes/bm25_chunk_ids.json"],
        "description": "Builds BM25Okapi keyword index for exact HTS code matching.",
    },
    "graph": {
        "label": "Build knowledge graph",
        "script": "step7_knowledgegraph.py",
        "outputs": [
            "indexes/hts_knowledge_graph.graphml",
            "indexes/hts_knowledge_graph.gpickle",
        ],
        "description": "Constructs HTS hierarchy graph, persists as GraphML + pickle.",
    },
    "verify": {
        "label": "Smoke-test HybridRetriever",
        "script": None,  # Runs in-process; no subprocess.
        "outputs": [],
        "description": "Loads the retriever and runs 3 sample queries to confirm correctness.",
    },
}

# Canonical stage order.
STAGE_ORDER = ["collect", "chunk", "faiss", "bm25", "graph", "verify"]

# Verification queries used in the 'verify' stage.
# Each entry is (query_text, chapter_filter_or_None, prefix_filter_or_None)
VERIFY_QUERIES = [
    ("purebred breeding horses import classification", None, None),
    ("0101.21.00.10", None, None),
    ("cotton t-shirts women general duty rate", "61", None),
]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_SEP = "=" * 70
_SEP_THIN = "-" * 70


def _fmt_size(path: Path) -> str:
    """Return human-readable file size, or '(missing)' if file absent."""
    if not path.exists():
        return "(missing)"
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def _print_banner():
    print()
    print(_SEP)
    print("  HTS RAG  -  End-to-End Pipeline")
    print(_SEP)


def _stage_header(name: str, label: str):
    print(f"\n>  Stage '{name}' -- {label}")


def _ok(msg: str):
    print(f"  [OK]  {msg}")


def _warn(msg: str):
    print(f"  [!!]  {msg}")


def _err(msg: str):
    print(f"  [XX]  {msg}")


def _info(msg: str):
    print(f"        {msg}")


# ---------------------------------------------------------------------------
# Core stage runner
# ---------------------------------------------------------------------------


def _all_outputs_exist(stage: str, root: Path) -> tuple:
    """Return (all_exist: bool, missing: list[str])."""
    outputs = STAGE_META[stage]["outputs"]
    if not outputs:
        return True, []
    missing = [o for o in outputs if not (root / o).exists()]
    return len(missing) == 0, missing


def run_stage(
    stage: str,
    root: Path,
    dry_run: bool = False,
    force: bool = False,
    skip_set=None,
) -> tuple:
    """
    Execute a single pipeline stage.

    Returns (status, elapsed_seconds).
    Status is one of: 'ok', 'skipped', 'failed', 'dry'.
    """
    meta = STAGE_META[stage]
    label = meta["label"]
    _stage_header(stage, label)

    # 1. Explicitly skipped by the caller.
    if skip_set and stage in skip_set:
        _warn(f"Stage '{stage}' explicitly skipped by --skip flag.")
        return "skipped", 0.0

    # 2. Dry-run mode: describe what would happen.
    if dry_run:
        script = meta["script"]
        if script:
            _info(f"[dry-run] Would run: python {script}")
        else:
            _info("[dry-run] Would run in-process verify stage.")
        outputs = meta["outputs"]
        if outputs:
            _info(f"[dry-run] Expected outputs: {', '.join(outputs)}")
        return "dry", 0.0

    t0 = time.perf_counter()

    # 3. Smart-skip: output files already present and --force not set.
    all_exist, missing = _all_outputs_exist(stage, root)
    if all_exist and not force and stage != "verify":
        for out in meta["outputs"]:
            p = root / out
            _warn(f"'{out}' already exists ({_fmt_size(p)}). Skipping.")
        elapsed = time.perf_counter() - t0
        return "skipped", elapsed

    # 4. Actually run the stage.
    script = meta["script"]
    if script is None:
        # In-process verify stage.
        status = _run_verify(root)
    else:
        script_path = root / script
        if not script_path.exists():
            _err(f"Script not found: {script_path}")
            return "failed", time.perf_counter() - t0

        _info(f"Running: python {script}")
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(root),
                check=False,
            )
            if result.returncode != 0:
                _err(f"Script exited with code {result.returncode}.")
                status = "failed"
            else:
                _ok("Script completed successfully.")
                status = "ok"
        except Exception as exc:
            _err(f"Unexpected error running {script}: {exc}")
            status = "failed"

    elapsed = time.perf_counter() - t0
    _info(f"Stage '{stage}' finished in {_fmt_elapsed(elapsed)}.")
    return status, elapsed


# ---------------------------------------------------------------------------
# In-process verification stage
# ---------------------------------------------------------------------------


def _run_verify(root: Path) -> str:
    """
    Load HybridRetriever (if indexes exist) and run smoke-test queries.
    Returns 'ok' or 'failed'.
    """
    # Ensure project root is importable.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Change CWD so relative index paths in HybridRetriever resolve correctly.
    original_cwd = os.getcwd()
    os.chdir(str(root))
    retriever = None
    try:
        from step5_retriever import HybridRetriever  # type: ignore

        t_load = time.perf_counter()
        retriever = HybridRetriever()
        load_elapsed = time.perf_counter() - t_load
        _ok(
            f"HybridRetriever loaded in {_fmt_elapsed(load_elapsed)} "
            f"({retriever.faiss_index.ntotal} vectors indexed)."
        )
    except ImportError as exc:
        _err(f"Could not import HybridRetriever: {exc}")
        os.chdir(original_cwd)
        return "failed"
    except FileNotFoundError as exc:
        _err(
            f"Index files not found: {exc}. "
            "Run the faiss and bm25 stages first."
        )
        os.chdir(original_cwd)
        return "failed"
    except Exception as exc:
        _err(f"Failed to initialise HybridRetriever: {exc}")
        os.chdir(original_cwd)
        return "failed"

    all_passed = True
    try:
        for query, chapter, prefix in VERIFY_QUERIES:
            label = f'"{query}"'
            if chapter:
                label += f" (chapter={chapter})"
            print(f"\n  Query: {label}")
            print(f"  {_SEP_THIN}")
            try:
                results = retriever.hybrid_search(
                    query, top_k=3, chapter=chapter, prefix=prefix
                )
                if not results:
                    _warn("No results returned.")
                    all_passed = False
                else:
                    for rank, r in enumerate(results, 1):
                        htsno = r["metadata"].get("htsno", "-")
                        desc = r["metadata"].get("description", "-")
                        score = r["rrf_score"]
                        print(
                            f"  Rank {rank} | RRF {score:.4f} "
                            f"| {htsno:<20} | {desc}"
                        )
            except Exception as exc:
                _err(f"Query failed: {exc}")
                all_passed = False
    finally:
        os.chdir(original_cwd)

    if all_passed:
        _ok("All verification queries completed successfully.")
        return "ok"
    else:
        _warn("Some verification queries failed or returned no results.")
        return "failed"


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_pipeline(
    stages,
    root: Path,
    dry_run: bool = False,
    force: bool = False,
    skip_set=None,
    stop_on_failure: bool = False,
) -> dict:
    """
    Run the specified stages in order.

    Returns dict of {stage: (status, elapsed)} for each attempted stage.
    """
    _print_banner()
    if dry_run:
        print("  Mode           : DRY RUN (no files will be changed)")
    print(f"  Stages to run  : {', '.join(stages)}")
    print(f"  Force rebuild  : {'yes' if force else 'no'}")
    print(f"  Project root   : {root}")

    results = {}
    previous_failed = False

    for stage in stages:
        # If a prior stage failed and stop_on_failure is set, mark as skipped.
        if previous_failed and stop_on_failure:
            results[stage] = ("skipped_dep", 0.0)
            continue

        status, elapsed = run_stage(
            stage,
            root=root,
            dry_run=dry_run,
            force=force,
            skip_set=skip_set,
        )
        results[stage] = (status, elapsed)

        if status == "failed":
            previous_failed = True

    # --- Summary table ---
    print(f"\n{_SEP}")
    print("  Pipeline Summary")
    print(_SEP)

    total_elapsed = sum(e for _, e in results.values())
    any_failed = False
    for stage in stages:
        st, elapsed = results.get(stage, ("?", 0.0))
        icon = {
            "ok": "[OK]",
            "skipped": " -- ",
            "dry": " -- ",
            "failed": "[XX]",
            "skipped_dep": " -- ",
        }.get(st, " ?? ")
        note = ""
        if st == "failed":
            note = "  <- FAILED"
            any_failed = True
        elif st == "skipped":
            note = "  (already built)"
        elif st == "skipped_dep":
            note = "  (skipped due to earlier failure)"
        elif st == "dry":
            note = "  (dry-run)"
        t = _fmt_elapsed(elapsed) if elapsed > 0 else ""
        print(f"  {icon}  {stage:<12} {t:<10} {note}")

    print()
    print(f"  Total wall-clock time: {_fmt_elapsed(total_elapsed)}")
    if dry_run:
        print("  Dry run complete. No files were modified.")
    elif any_failed:
        print("  Pipeline finished with errors. [XX]")
    else:
        print("  Pipeline completed successfully. [OK]")
    print(_SEP)

    return results


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------


def cmd_status(root: Path):
    """Print the existence and size of all pipeline artifact files."""
    print()
    print(_SEP)
    print("  HTS RAG  -  Pipeline Status")
    print(_SEP)
    print(f"  Project root: {root}")
    print()

    for stage in STAGE_ORDER:
        meta = STAGE_META[stage]
        outputs = meta["outputs"]
        label = meta["label"]
        if not outputs:
            print(f"  [--] [{stage:<8}]  {label}")
            print("              No artifact files (in-process stage).")
            continue

        all_exist = all((root / o).exists() for o in outputs)
        status_icon = "[OK]" if all_exist else "[--]"
        print(f"  {status_icon} [{stage:<8}]  {label}")
        for out in outputs:
            p = root / out
            size_str = _fmt_size(p)
            exists = "+" if p.exists() else "-"
            print(f"      {exists}  {out:<45} {size_str}")
    print()


# ---------------------------------------------------------------------------
# search command
# ---------------------------------------------------------------------------


def cmd_search(root: Path, query: Optional[str]):
    """Load HybridRetriever and run an interactive or one-shot search."""
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    original_cwd = os.getcwd()
    os.chdir(str(root))
    try:
        from step5_retriever import HybridRetriever  # type: ignore

        print("Loading HybridRetriever...")
        retriever = HybridRetriever()
    except Exception as exc:
        print(f"Error loading retriever: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        os.chdir(original_cwd)

    def _do_search(q: str):
        os.chdir(str(root))
        try:
            results = retriever.hybrid_search(q, top_k=5)
        finally:
            os.chdir(original_cwd)
        print(f"\nResults for: '{q}'")
        print(_SEP_THIN)
        if not results:
            print("  (no results)")
        for rank, r in enumerate(results, 1):
            htsno = r["metadata"].get("htsno", "-")
            desc = r["metadata"].get("description", "-")
            chapter = r["metadata"].get("chapter", "-")
            # Extract General Duty Rate from the raw text block.
            general = None
            for line in r["text"].splitlines():
                if line.startswith("General Duty Rate:"):
                    general = line.split(":", 1)[1].strip()
                    break
            print(
                f"  Rank {rank} | RRF {r['rrf_score']:.4f} "
                f"| Ch.{chapter} | {htsno:<22} | {desc}"
            )
            if general:
                print(f"            General Duty Rate: {general}")
        print(_SEP_THIN)

    if query:
        _do_search(query)
    else:
        print("HTS RAG Interactive Search  (type 'quit' to exit)")
        print(_SEP_THIN)
        while True:
            try:
                q = input("\nQuery> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                continue
            if q.lower() in {"quit", "exit", "q"}:
                break
            _do_search(q)


# ---------------------------------------------------------------------------
# run-step command
# ---------------------------------------------------------------------------

_STEP_MAP = {
    1: "collect",
    2: "chunk",
    3: "faiss",
    4: "bm25",
    5: "verify",
    6: "graph",
}


def cmd_run_step(step_num: int, root: Path, force: bool = False, dry_run: bool = False):
    """Run a single pipeline step by number (1-6)."""
    stage = _STEP_MAP.get(step_num)
    if stage is None:
        print(
            f"Unknown step number: {step_num}. Valid range: 1-6.",
            file=sys.stderr,
        )
        sys.exit(1)
    run_pipeline([stage], root=root, dry_run=dry_run, force=force)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="step6_pipeline.py",
        description="HTS RAG end-to-end pipeline orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python step6_pipeline.py status
  python step6_pipeline.py build --dry-run
  python step6_pipeline.py build --skip collect chunk
  python step6_pipeline.py build --only faiss bm25 verify
  python step6_pipeline.py build --force
  python step6_pipeline.py search "cotton t-shirts women"
  python step6_pipeline.py search
  python step6_pipeline.py run-step 3
""",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # status
    sub.add_parser("status", help="Show which artifact files exist and their sizes.")

    # build
    build_p = sub.add_parser("build", help="Run pipeline stages (default: all).")
    build_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without executing anything.",
    )
    build_p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild all stages even if output files already exist.",
    )
    build_p.add_argument(
        "--skip",
        nargs="+",
        metavar="STAGE",
        default=[],
        choices=STAGE_ORDER,
        help=f"Stages to skip. Choices: {', '.join(STAGE_ORDER)}",
    )
    build_p.add_argument(
        "--only",
        nargs="+",
        metavar="STAGE",
        default=[],
        choices=STAGE_ORDER,
        help="Run only these stages (overrides --skip).",
    )
    build_p.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Halt the pipeline at the first stage failure.",
    )

    # search
    search_p = sub.add_parser(
        "search",
        help="Run an interactive or one-shot hybrid search.",
    )
    search_p.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Query string. Omit for interactive mode.",
    )

    # run-step
    step_p = sub.add_parser("run-step", help="Run a single step by number (1-6).")
    step_p.add_argument(
        "step_num", type=int, metavar="N", help="Step number (1-6)."
    )
    step_p.add_argument("--force", action="store_true")
    step_p.add_argument("--dry-run", action="store_true")

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv=None):
    args = _parse_args(argv)
    root = _HERE

    if args.command == "status":
        cmd_status(root)

    elif args.command == "build":
        if args.only:
            stages = [s for s in STAGE_ORDER if s in args.only]
        else:
            stages = list(STAGE_ORDER)

        skip_set = set(args.skip) if args.skip else set()

        run_pipeline(
            stages=stages,
            root=root,
            dry_run=args.dry_run,
            force=args.force,
            skip_set=skip_set,
            stop_on_failure=args.stop_on_failure,
        )

    elif args.command == "search":
        cmd_search(root, args.query)

    elif args.command == "run-step":
        cmd_run_step(
            step_num=args.step_num,
            root=root,
            force=args.force,
            dry_run=args.dry_run,
        )

    else:
        # No subcommand: default to status.
        cmd_status(root)


if __name__ == "__main__":
    main()
