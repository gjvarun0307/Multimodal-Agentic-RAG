"""Real end-to-end retrieval-only eval.harness run against the actual
corpus and Milvus collection -- no LLM key needed, since --retrieval-only
never constructs one (PROJECT_SPEC.md §7 Phase 2: "Retrieval-only mode
makes zero LLM calls"). Skipped when the corpus isn't available locally,
matching tests/test_smoke.py's convention.

This is the one real (non-monkeypatched) check that eval.harness actually
works against the live index -- tests/test_harness.py covers the pure
logic and the LLM-dependent full-mode path with fakes.
"""

from pathlib import Path

import pytest

from eval.harness import run_eval

REPO_ROOT = Path(__file__).resolve().parent.parent
PARSED_MD_DIR = REPO_ROOT / "artifacts" / "parsed_md"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _corpus_present() -> bool:
    return PARSED_MD_DIR.exists() and any(PARSED_MD_DIR.glob("*.md"))


pytestmark = pytest.mark.skipif(
    not _corpus_present(),
    reason="requires the local corpus (artifacts/parsed_md/) and a built Milvus collection",
)


def test_retrieval_only_fast_split_runs_against_real_corpus(tmp_path):
    results, out_path = run_eval(
        config_path=DEFAULT_CONFIG,
        split="fast",
        retrieval_only=True,
        results_dir=tmp_path,
    )

    assert out_path.exists()
    assert results["backend"] == "retrieval-only"
    assert results["warmup_excluded"] is True
    assert results["n_items"] > 0

    retrieval = results["metrics"]["retrieval"]
    assert retrieval["n_items_scored"] > 0
    # Real hybrid search against the real index should find *something* --
    # not asserting a specific recall value here (that's a noise-floor/
    # baseline concern, Phase 3+), just that the pipeline produced real,
    # non-degenerate numbers.
    assert 0.0 <= retrieval["stage1"]["recall@10"] <= 1.0
    assert retrieval["stage2"]["recall@5"] is not None

    for item in results["per_item"]:
        assert isinstance(item["retrieved_chunk_ids"], list)
