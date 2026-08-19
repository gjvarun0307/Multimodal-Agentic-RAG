# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

**Source of truth:** `PROJECT_SPEC.md` (frozen). This file is the *living* companion —
distilled working instructions, current repo state, and the current phase's checklist.
When this file and `PROJECT_SPEC.md` disagree, `PROJECT_SPEC.md` wins; fix this file to
match, not the other way around. Update this file at phase boundaries (spec §7) — track
the current phase's steps here and check them off as they land; don't restate finished
phases in prose, git history and the committed docs already have that.

## What this project is now

A working single-shot LangGraph agentic RAG demo (adaptive routing + Self-RAG
correction, hybrid Milvus retrieval, BYOK LLM) is becoming a **measured, deployed,
CI-gated evaluation project**. Full architecture, phase plan, and rationale live in
`PROJECT_SPEC.md` — read it before starting any phase, it is long and dense by design
and this file does not restate it.

**Zero new user-facing features.** Every change must serve measurement, reliability,
deployment, or reproducibility (spec §2.1). Off-topic ideas go in `docs/BACKLOG.md`
(not yet created), not into the codebase.

## Phase state

0 Unblock (done) → 1 Golden set (done) → 2 Metrics harness (done) → 3 Noise floor
(done) → **4 Eval-as-CI (in progress)** → 5 Deploy & observability → 6 Ablations &
failure taxonomy → 7 README rewrite.

Full detail per phase in `PROJECT_SPEC.md` §7. Completed-phase facts worth carrying
forward live in "Facts carried forward" below and in the committed docs they
reference — not restated here as narrative.

## Phase 4 — Eval-as-CI (current)

Spec §7 Phase 4's own text has two internal conflicts, resolved with the user
2026-08-16 before build started — both decisions, not open questions:

1. **Corpus delivery to CI: private Hugging Face Dataset.** CI has no
   corpus/index today and the corpus is deliberately not committed
   (`artifacts/SOURCES.md` — license restrictions per paper).
   `deploy/fetch_corpus.py` / `deploy/build_ingest_artifacts.py` (spec's
   intended long-term fix) are Phase 5 deliverables that don't exist yet.
   Upload `milvus.db` + `artifacts/parsed_md/` once to a *private* HF
   Dataset; CI fetches it via an `HF_TOKEN` secret. Re-upload whenever
   corpus/chunking/embedding model changes — it's a build artifact, not
   source of truth (`artifacts/metadata.jsonl` / `corpus_seed.csv` stay
   canonical).
2. **Fast tier = zero LLM calls, literally.** Spec's tier table says fast =
   "retrieval + structured, no judge" but its acceptance criteria say "zero
   LLM calls" — structured-output metrics call the generation LLM, which
   risks real Groq-quota cost on every push. Fast tier =
   `eval.harness --split fast --retrieval-only` +
   `test_chunk_id_determinism.py` only. Structured-output metrics move into
   the nightly/labeled full tier with the judge, where cost is already
   budgeted.

**Gate thresholds** (from `docs/NOISE_FLOOR.md`, fast split, n=4 —
deliberately overrides spec's literal gate table, which blocks on
`faithfulness` and `refusal_accuracy`; Phase 3 found those judge metrics n=1
or unusable σ, so they're informational-only until a dedicated judge-only
noise-floor pass exists):

| Metric | Threshold | Action |
|---|---|---|
| `chunk_id_determinism` | any failure | **block** |
| `retrieval.stage1.recall@10` | drop > 0.045 | **block** |
| `router.accuracy` | drop > 0.03 | warn |
| `structured.aggregate.validity_rate` | any drop from 1.000 | warn |
| `correction.fire_rate` | drop > 0.07 | warn (full tier only) |
| `system.warm_p95_ms` | increase > 20% | warn |
| `faithfulness`, `citation_precision`, `refusal_accuracy`, `answer_correctness` | — | informational only, no gate |

**Already usable, no new build needed:** `eval.harness --split fast
--retrieval-only` (live-verified Phase 2), `TAVILY_MODE=replay` (no live key
needed), `tests/test_chunk_id_determinism.py` (Phase 0) — CI should invoke
it as a real pytest job, not reimplement the check.

**Real timing finding (2026-08-18), overrides the fast-tier "< 3 min"
budget:** a live `eval.harness --split fast --retrieval-only` run against
the real 39-item split, on CPU (no GPU), took **~12 minutes wall clock** end
to end — zero LLM calls as designed, but real BGE-M3 dense+sparse query
embedding and `bge-reranker-v2-m3` cross-encoder scoring per item is
genuinely slow on CPU. GitHub-hosted Actions runners have no GPU either, so
this isn't a local-machine-only artifact. **Decided with the user
2026-08-18: accept ~12–15 min as the real fast-tier budget, keep the full
39-item split (preserves Phase 3's noise-floor thresholds, which were
computed against exactly this split), document "< 3 min" as an unrealistic
spec target rather than shrinking the item count.** Cost is still $0 (public
repo, unlimited Actions minutes) — only PR feedback latency changes, not
correctness or noise-floor validity. `.github/workflows/fast-eval.yml`'s job
`timeout-minutes: 30` reflects this (bumped from an initial 15 after a real
Actions run hit ~15m15s total and got killed by the timeout, not a real
failure — every step through the harness run itself succeeded).

**Build checklist** (check off as each lands; repo confirmed public
2026-08-18 — unlimited free Actions minutes, no action needed):

- [x] Upload `milvus.db` + `artifacts/parsed_md/` to a private HF Dataset —
      done 2026-08-18, `gjvarun0307/arag-eval-corpus` (private). CI needs a
      separate **read-only** fine-grained `HF_TOKEN` scoped to this repo
      (not the write-scoped `upload-bundle-token` used to create it).
- [x] `.github/workflows/fast-eval.yml` — every push/PR, retrieval-only +
      determinism. Budget ~13–15 min (see timing finding above), not spec's
      literal < 3 min; $0 either way. **Fully green end-to-end in a real
      Actions run, 2026-08-18** (run 32119217309, `timeout-minutes: 30`) —
      that run predates the `eval.gate` wiring below, i.e. it's proof the
      harness/determinism steps work in real CI, not the gate step.
      **`eval.gate` wired in, 2026-08-18** (same day, follow-up commit):
      determinism step gets `continue-on-error: true` +
      `id: determinism` so a determinism failure doesn't kill the job
      before a gate report exists — `eval.gate`'s own exit code (1 on
      block) is what fails the job now, via `--determinism-passed
      ${{ steps.determinism.outcome == 'success' && 'true' || 'false'
      }}`. The harness run's own results JSON is picked out of
      `eval/results/*.json` by `ls -t` (that dir has 8 pre-existing
      committed historical runs, so it isn't the only file there — a fresh
      checkout gives them all an earlier, uniform mtime than the just-written
      one). Diff table goes to `$GITHUB_STEP_SUMMARY` (visible on the Actions
      run page, no new permissions needed) and both `gate_report.json` +
      `gate_comment.md` are uploaded alongside the existing artifacts.
      **Not yet posting an actual PR-comment via the GitHub API** — that
      needs `pull-requests: write` and `actions/github-script`, a step up
      in blast radius (visible to any PR viewer, not just Actions-run
      viewers), deliberately held off pending explicit sign-off.
      **Verified in real CI, 2026-08-18** — the "deliberately-broken
      config" checklist item below both proved the wiring and caught a
      real bug in it (`tee` swallowing `eval.gate`'s exit code without
      `set -o pipefail`, fixed in `9357eec`); a second real run after the
      fix shows the job/check correctly going to `failure` on a genuine
      BLOCK.
- [x] `.github/workflows/full-eval.yml` — done 2026-08-18. `run-full-eval`
      label trigger only for now, **not nightly** — decided with the user:
      Groq's free-tier budget has already been exhausted twice from manual
      runs, so an unattended nightly 145-item run shouldn't go live before
      a single labeled run has actually succeeded in CI. `schedule:` cron
      is written but commented out, ready to enable once that track record
      exists. Needs `LLM_API_KEY`/`JUDGE_API_KEY` repo secrets (user
      confirmed already added) in addition to `fast-eval.yml`'s `HF_TOKEN`
      — `TAVILY_MODE=replay` is hardcoded and no `TAVILY_API_KEY` secret is
      set at all (invariant 12, defense in depth). Mirrors
      `fast-eval.yml`'s setup steps and already includes the
      `set -o pipefail` fix from the start. Gates against the same
      `eval/baselines/main.json` (fast/retrieval-only) — recall@10 still
      diffs for real, every full-tier-only metric reports "no baseline"
      until a full-tier baseline exists (separate, not-yet-scoped work).
      **Unverified in real CI** — no labeled PR has triggered it yet;
      `timeout-minutes: 45` is an unverified guess (spec says < 25 min,
      but full-tier adds real per-item LLM/judge network latency on top of
      fast-eval's own ~12-15 min retrieval time for a smaller item count).

      **`LLM_PROVIDER` switched `groq` -> `nvidia_nim`, 2026-08-18** — the
      user's local `api_keys.json` was already `nvidia_nim` /
      `meta/llama-3.1-8b-instruct` for generation (discovered mid-session,
      not what this workflow originally assumed); CI now matches. The
      `LLM_API_KEY` secret was reused (its value swapped to an NVIDIA NIM
      key, same secret name, user confirmed) rather than renamed, so no
      other workflow change was needed. `JUDGE_API_KEY` is unaffected and
      stays a Groq key — invariant 11 pins the judge to Groq
      (`llama-3.3-70b-versatile`) independent of `LLM_PROVIDER`, so this
      move doesn't touch the judge's own Groq TPD budget, which is now the
      *only* Groq consumer in this workflow (generation no longer competes
      with it). Real math from a committed 39-item full run
      (`eval/results/20260815T054051Z_fb38c61.json`, 76 judge calls total
      across faithfulness/citation/correctness/refusal) extrapolates to
      ~283 judge calls for the full 145-item split — plausibly still
      several times the judge's 100k free-tier TPD in one run (token
      counts aren't recorded per-call in that file, so this is an
      order-of-magnitude estimate, not a measured number). If a labeled
      run does exhaust it mid-run, the rate-limit-distinction work above
      means the gate reports those judge metrics `rate_limited`, not a
      fabricated FAIL — but full judge coverage still isn't guaranteed in
      one shot. `--split dev` (30 items) remains the recommended cheap
      first look to get a real per-call token number before committing to
      a full 145-item run.

      **Deliberately not yet triggered, 2026-08-18 — user is holding off,
      budget-sensitive** (same Groq 100k TPD constraint as above). Resume
      point for a future session: this is the very next thing to do once
      the user says go. Three ways to trigger, decided but not yet acted
      on:
      - **Option A (recommended) — real PR + label.** Push any branch with
        a real change, open a PR against `main`, add the label
        `run-full-eval` (create it in the PR sidebar if it doesn't exist
        yet — a plain label, no special config). The job's
        `if: github.event.label.name == 'run-full-eval'` guard means only
        that exact label fires it. Watch at
        `.../actions/workflows/full-eval.yml`.
      - **Option B — manual `workflow_dispatch` button.** Not wired in
        yet — `full-eval.yml` has no `workflow_dispatch:` trigger. A
        one-line addition if the user wants a no-PR "Run workflow" button
        instead of Option A.
      - **Option C — run locally, zero Actions cost, same Groq cost.**
        `python -m eval.harness --config configs/default.yaml --split
        full` then `python -m eval.gate --run
        eval/results/<new_run_id>.json --baseline
        eval/baselines/main.json`. `--split dev` (30 items, same code
        path) is the cheap first look before committing to the full
        145-item `--split full` run.
      This first run is also what will finally verify (a) real full-tier
      wall-clock time against the 45-min timeout guess and (b) whether
      `--split full` actually completes end to end in the CI environment
      at all — neither has been checked yet.
- [x] `.github/workflows/lint.yml` — `ruff check .` + `pytest`. **Fully
      green in a real Actions run, 2026-08-18** (run 32119217301).

**Known defect found while wiring lint.yml (2026-08-18), not fixed —
`app.py` (Streamlit) cannot be imported.** It still imports the deleted
top-level `config.py`/`hybrid_database.py`/`parse.py` (removed in `381c51d`,
"move core modules into src/, delete config.py shim") and reimplements its
own `load_runtime()`/`build_graph()`/`run_query()` instead of calling
`src.runtime.get_runtime()` — directly contradicting invariant 14 and
`runtime.py`'s own docstring. It's a partial migration, not just stale: the
top ~450 lines are the pre-refactor implementation, but further down
(`main()`) references `runtime.database`/`runtime.embedding_model` and
`MAX_CHAT_TURNS` as if the migration had already happened — both are
genuinely undefined names (`ruff` F821, confirmed real bugs not style).
`src/api.py` (FastAPI) is unaffected and confirmed working (`test_api.py`,
`test_smoke.py` both pass against the real `get_runtime()` path). Separately,
this file still has a full upload UI (`st.file_uploader`, `ingest_uploaded_pdf`)
despite CLAUDE.md's "Repo facts" claiming uploads were rejected and no
upload UI exists — that claim is stale and needs reconciling once this file
is actually fixed. **Decided with the user 2026-08-18: excluded from `ruff`
(`pyproject.toml` `[tool.ruff] exclude`) rather than rewritten now — the
rewrite (get_runtime()-based, plus resolving the upload-UI question) is
separate follow-up work, not folded into Phase 4 CI setup.**

**Same disease, different file: `src/parse.py` also imports deleted
top-level modules** (`from config import config_parse`, `from helper import
clean_json_text` — should be `.configuration`/`.helper`, matching every
other file under `src/`). Not caught by `ruff` (no import-resolution rule
selected) or by the full pytest run (nothing in `tests/` currently imports
`src.parse` — it's ingest-only, never on the query path per its own module
docstring, so this has been silently dormant). Not fixed or excluded here
since it isn't blocking anything today; flagged so it isn't a surprise
whenever ingest/re-parsing is next touched (Phase 5's `deploy/fetch_corpus.py`
work is the likely trigger).
- [x] Gate-comparison script — `eval/gate.py` (`python -m eval.gate --run
      <results.json> --baseline eval/baselines/main.json
      [--determinism-passed true|false] [--out report.json]`), done
      2026-08-18. Diffs a results JSON against the baseline per the gate
      table above, emits per-metric pass/warn/block, renders the PR-comment
      diff table (spec §7 example). Reuses `eval.noise_floor.
      flatten_numeric_metrics` rather than reimplementing flattening.
      Metrics absent from the *current* run (e.g. every full-tier-only row
      on a `--retrieval-only` fast run) report "not run this tier", never a
      fabricated pass (invariant 15); metrics absent from the baseline
      report "no baseline" rather than crashing -- expected until the next
      checklist item lands. `chunk_id_determinism` is folded in via
      `--determinism-passed` since it's pytest's own gate
      (`tests/test_chunk_id_determinism.py`), not part of the harness JSON.
      14 unit tests in `tests/test_gate.py`; live-verified against real
      `eval/results/*.json` files -- a real +0.022 recall@10 delta passes,
      an injected -0.10 delta blocks with exit 1. Full `pytest` suite (175
      tests) and `ruff check .` both clean after adding it. Does **not**
      yet solve rate-limit-vs-regression distinction (next-but-one item
      below) -- flagged as a known gap in the module's own docstring, not
      silently handled.
- [x] `eval/baselines/main.json` — done 2026-08-18. A verbatim copy of a
      real `eval.harness --config configs/default.yaml --split fast
      --retrieval-only` run's results JSON, never hand-written (invariant
      7): `run_id=20260818T184206Z_cf44a44`, `git_sha=cf44a44` (current
      `main` HEAD at the time), `split=fast`, `backend=retrieval-only`,
      `n_items=39`, `retrieval.stage1.recall@10=0.6304`. Self-diffed
      against `eval/gate.py` to confirm a zero-delta pass end to end.
      Baseline updates after this are PR-only with written justification,
      never automatic.
- [x] Rate-limit/quota errors rendered distinctly from real regressions, in
      both gate logic and PR comment (a Groq 429 must never show as ❌ FAIL)
      -- done 2026-08-18. Every LLM call in this project (generation, judge,
      router) goes through `langchain_openai.ChatOpenAI` against a
      provider's OpenAI-compatible endpoint (`src/configuration.py`), so a
      429 always surfaces as `openai.RateLimitError` regardless of which
      provider is configured -- `src/helper.py`'s `is_rate_limit_error()`
      detects it by walking the exception's `__cause__`/`__context__`
      chain. `src/agent.py`'s top-level graph-exception catch now tags a
      rate-limited item `"rate_limited"` instead of the generic
      `"graph_execution_error"`, which flows into
      `system.error_rate_by_stage.error_rate_by_tag.rate_limited`
      automatically (no new plumbing -- that aggregation already existed).
      `eval/judge.py` gained `JudgeRateLimitError(JudgeGradingError)`,
      raised instead of the base class when `is_rate_limit_error()` is
      true; `eval/harness.py` counts these per judge dimension
      (faithfulness/citation_precision/answer_correctness/refusal_accuracy)
      and merges `n_rate_limited` alongside each dimension's existing
      `n_scored` in the results JSON. `eval/gate.py` gained a `rate_limited`
      status (⏳, never counts toward WARN/BLOCK in `overall_verdict`):
      overrides a would-be WARN/BLOCK on the three LLM-driven full-tier
      gated metrics (`router.accuracy`, `structured.validity_rate`,
      `correction.fire_rate` -- marked `rate_limit_sensitive=True`, unlike
      `recall@10`/`system.warm_p95_ms` which aren't LLM-driven and still
      gate normally during a rate-limited run) when the run's
      `error_rate_by_tag.rate_limited` is nonzero; overrides
      `insufficient_data` on the four judge-scored info metrics with the
      more specific `rate_limited` reason when that dimension's
      `n_rate_limited` is nonzero. 7 new unit tests (`tests/test_gate.py`,
      `tests/test_judge.py`) verify the identical numeric drop renders WARN
      on a clean run vs `rate_limited` on a quota-constrained one, and that
      a non-rate_limit_sensitive metric still gates normally either way.
      Full `pytest` (182 tests) and `ruff check .` both clean.
- [x] A deliberately-broken retrieval config, run through the fast workflow,
      proving the gate actually blocks a test PR (acceptance criterion) —
      done 2026-08-18. Test PR from branch `test/gate-block-proof`
      (`search_limit: 50 -> 1`, collapsing stage1 recall@10 0.630 -> 0.261,
      commit `5adc7b2`) against `eval/baselines/main.json`. **First run
      (`32177680250`) exposed a real bug, not just proved the concept:**
      the "Eval gate" step's `python -m eval.gate ... | tee gate_comment.md
      >> $GITHUB_STEP_SUMMARY` reported job conclusion `success` even
      though the step summary correctly showed "❌ Eval gate: BLOCKED" --
      GitHub Actions' default `bash` shell for `run:` steps does not set
      `pipefail`, so the step's exit status was `tee`'s (always 0),
      silently swallowing `eval.gate`'s exit 1. Fixed on `main`
      (`9357eec`, `set -o pipefail` added before the pipeline) and merged
      into the test branch to re-trigger; re-run (`32180138001`)
      correctly shows job/step conclusion `failure`. Test PR closed
      without merging, `test/gate-block-proof` branch deleted -- the
      config change was never meant to land, only the pipefail fix
      (already on `main`) is real product of this item.
- [x] Screenshot a real blocked-merge run for the Phase 7 README -- done
      2026-08-18, `imgs/ci_blocked_merge_run_32180138001.jpg`. Captured
      from the already-proven gate-block run (`32180138001`, see the item
      above) rather than triggering a new one -- shows job/step conclusion
      `Failure` and the "❌ Eval gate: BLOCKED" summary table
      (`recall@10` 0.630 → 0.261). Saved here now so Phase 7 doesn't need
      to re-trigger a blocked run just to get the image; not yet placed
      into an actual README (that's Phase 7's job).

**Deferred, not this phase:** `.github/workflows/keep-warm.yml` — no Space
exists until Phase 5, nothing to ping yet.

**Incident (2026-08-18): the corpus was actually committed to git and public
on `origin/main` for ~1 week, contradicting the "corpus is not committed"
claim this whole plan (and the private-HF-Dataset decision above) rests on.**
All 15 `artifacts/parsed_md/*.md` files were added in `e461362` (part of the
2026-08-11 reseed, not caused by any Phase 4 work) and were never actually
gitignored-in-practice — `.gitignore` listing a path doesn't untrack files
already committed before the rule existed. Found because `lint.yml`'s pytest
run didn't skip `test_harness_live.py`/`test_resolve_passages.py`'s
real-corpus tests on a fresh CI checkout, which should have been the tell
that the corpus was actually present there. **Remediated the same day:**
full-history scrub via `git filter-repo --path artifacts/parsed_md
--invert-paths --force` + force-push to `origin/main` (backup bundle taken
first, `~/Hecker/backups/multimodal-agentic-rag_full_repo_backup_2026-08-18.bundle`,
covers `main` + the local-only `archive/pre-reseed-2026-08-11` branch, which
was never on origin and wasn't otherwise touched). Current `main` tree and
all reachable history are clean (verified via GitHub's tree API). **Known,
accepted residual exposure, not further fixable from here:** the pre-scrub
commit SHA is still directly fetchable from GitHub until their scheduled GC
runs (force-push doesn't instantly purge unreachable objects), and anyone
who already cloned/forked before today keeps the old content regardless.
`data/raw_pdfs/*.pdf` was checked and confirmed never committed — this
incident was scoped to `parsed_md/` only.

## Facts carried forward from earlier phases

Non-obvious conventions later phases depend on. Full build logs for finished
phases are git history + `docs/NOISE_FLOOR.md` + `eval/judge_calibration/`,
not restated here.

**Golden set** (`eval/golden/golden_set.jsonl` 145 items + `dev_split.jsonl`
30 items, Phase 1, closed 2026-08-11): `gold_passages` (verbatim text +
`doc_id`) is the only ground truth — chunk IDs are always derived at eval
time, never stored (invariant 4). Resolved chunk IDs cache per config at
`eval/golden/resolved/<config_hash>.json`. Post-freeze edits are a `version`
bump, not a silent edit (invariant 2). Never tune against `golden_set.jsonl`
— `dev_split.jsonl` only (invariant 6). Most VLM figure captions in this
corpus are low quality (repeated "Blank Image" captions) — a Phase 6
negative-results finding, not a Phase 1 blocker.

**Eval harness** (`eval/harness.py`, `eval/metrics/*.py`, Phase 2, closed
2026-08-15/16): `eval/metrics/router.py` maps `expected_route: "refuse"`
items to `vectorstore` for router-accuracy purposes only
(`EXPECTED_ROUTE_FOR_ROUTER`) — actual refusal behavior is
`generation.py`'s `refusal_accuracy`. `eval/metrics/structured.py`'s
`InstrumentingLLM` wraps `llm_model` to surface structured-output parse
failures/retries as data; `silent_coercion_rate` is a best-effort heuristic
that returns `None` (not a fake number) when it can't be checked.
`retrieve_and_rerank_core` (`src/agent.py`, module-level) lets
`--retrieval-only` exercise the exact retrieval path the real graph uses.
`correction_improve_rate`/`degrade_rate` always return `None` — needs a
`GraphState` change to preserve pre-correction generation text, not yet
scoped. Judge is pinned to `llama-3.3-70b-versatile` on Groq
(`DEFAULT_JUDGE_MODEL`, `eval/judge.py`) — every `with_structured_output()`
call needs `method="function_calling"` (Groq's `json_schema` mode only
supports the gpt-oss family). `JUDGE_VERSION` (`"v1"`) must move in
lockstep with `configs/default.yaml`'s `judge_version` on any prompt change
(invariant 11). `TAVILY_MODE=replay` (default) reads
`eval/fixtures/tavily_v1.json`, keyed by sha256 of `(query, max_results,
topic)`; a residual ~10% miss rate on the fast split's `web_search`
fallback is expected, not a bug (LLM-rewritten query text varies slightly
run-to-run even at `temperature: 0`). `temperature` is a first-class
`config_rag()` field (`TEMPERATURE` env var), `None` by default (prod
unaffected), pinned to `0` in `configs/default.yaml` (eval-only).

**Groq free-tier 100k TPD token budget is a real, already-hit constraint**
(hit twice: Phase 2, Phase 3) — a handful of consecutive full-mode
`eval.harness` runs in one day exhausts it; `JudgeGradingError` skips
gracefully rather than crashing, visible as smaller `n_scored`, not a
quality regression.

**Noise floor** (`eval/noise_floor.py`, Phase 3, closed 2026-08-16, full
write-up `docs/NOISE_FLOOR.md`): fast split (39 items), n=4 not 5 — same
Groq-budget reasoning, documented deviation. `--existing-results` resume
flag seeds the aggregate from already-completed run JSONs so a killed pass
doesn't waste spent quota. Headline non-judge metrics have clean n=4
coverage (feeds the Phase 4 gate table above); judge-scored metrics
(`faithfulness`, `citation_precision`) got n=1 — no variance computable,
hence informational-only in the gate table, not a papered-over number
(invariant 16).

## Non-goals (do not implement, park in `docs/BACKLOG.md`)

New retrieval strategies (HyDE/RAPTOR/GraphRAG/ColBERT), multi-document scoping,
MCP integration, streaming tokens, conversation persistence, Docling migration,
image-input queries, UI redesign, any new agent hop/tool. Three exceptions are in
scope: defect fixes, the vLLM backend as a *measurement instrument* (spec §4A, not
a deployed feature), and CPU-viable reranking (deploy target has no GPU).

## Global invariants (`PROJECT_SPEC.md` §8)

1. No new features (see Non-goals above).
2. Corpus is frozen — any change is an eval-set version bump + re-verification.
3. Chunk IDs are deterministic; rebuilds from identical sources produce identical IDs, enforced in CI.
4. `gold_passages` is ground truth; chunk IDs are always derived, never hand-edited.
5. `doc_id` is immutable once assigned.
6. Never tune against the main golden set — dev split only.
7. Every published number traces to a committed results JSON (git SHA, config, golden-set version, judge version, Tavily fixture version, backend).
8. Retrieval and generation metrics are always reported separately.
9. Warm and cold latency are always reported separately, never blended.
10. Ablations are config-only — a code-edit ablation means config is incomplete.
11. Judge is pinned (Groq) and independent of the generation model; prompt changes force a re-baseline.
12. Tavily replays from frozen fixtures in every eval/CI run — never live.
13. Precached demo output (Tier 1) is always labeled as precached and recorded from real runs, never hand-written.
14. Streamlit and FastAPI both call `get_runtime()` in-process — no self-HTTP hop.
15. Silent fallbacks are forbidden — disabled reranker, skipped model, rate limit, cache miss: all loud, logged, surfaced in results.
16. Negative results get published — a component that doesn't earn its cost is a finding, not an embarrassment.

## Repo facts

- **Corpus is not committed to git** (`data/raw_pdfs/*.pdf`,
  `artifacts/parsed_md/*.md`) — license restrictions per paper
  (`artifacts/SOURCES.md`). `artifacts/metadata.jsonl` and `corpus_seed.csv`
  stay committed. The corpus must exist locally (built via `parse.py`'s
  pipeline; `deploy/fetch_corpus.py` is planned, not built) before `pytest`
  runs fully or the index rebuilds. Corpus-dependent tests skip cleanly, not
  fail, when it's absent.
- **Module layout:** core modules under `src/` (`runtime.py`, `api.py`,
  `agent.py`, `hybrid_database.py`, `parse.py`, `configuration.py`,
  `helper.py`, `logging_utils.py`); `app.py`/`pages/1_Setup.py` at repo root.
- **Uploads are rejected** — corpus is frozen, `app.py` is *intended* to have
  no upload UI. **Stale as of 2026-08-18: `app.py` still has a full upload
  flow** (see Phase 4's `app.py` known-defect note) — this line describes
  the intended end state, not current code, until that file is fixed.

## Implementation notes for later phases

- **Config-only overrides:** `config_rag(overrides={...})` applies a dict on top of the
  resolved config (`defaults → api_keys.json → env → overrides`). This is the mechanism
  the eval harness and ablation runner (Phase 6) construct a run from — reach for this,
  not env vars.
- **Route decision isn't a `GraphState` field.** `query_router` is a conditional-edge
  selector, so its choice is never written into state. Use
  `agent.run_query_with_state()` — returns `(answer, final_state, trace_info)`, and
  `trace_info["node_sequence"][0]` is the route decision.
- **Milvus collection name is `arag_project_v2`** (bumped when chunk IDs went
  deterministic; an old `arag_project` collection may still exist locally alongside it).
- **Reranker default is `bge-reranker-v2-m3`** (CPU-viable) via
  `configuration.build_reranker()`; `bge-reranker-v2-gemma` stays available as a config
  value for the GPU ablation row.
- **vLLM ablation backend:** set `llm_provider=vllm` and `vllm_base_url` (a config
  field, not part of the static `LLM_PROVIDERS` entry) to point at a tunneled vLLM server.

## Commands

```bash
# setup
pip install -r requirements.lock      # exact reproducibility (or requirements.txt for direct deps only)

# dev
pytest
ruff check . && ruff format .

# corpus / metadata (requires data/raw_pdfs/*.pdf + artifacts/parsed_md/*.md locally)
python deploy/build_metadata.py --seed artifacts/corpus_seed.csv --out artifacts/metadata.jsonl
python -m src.hybrid_database          # (re)build ./milvus.db -- DESTRUCTIVE, drops + recreates arag_project_v2

# serve
uvicorn src.api:app --reload --port 8000
streamlit run app.py

# eval (golden set; requires corpus locally, see below)
python -m eval.resolve_passages         # writes eval/golden/resolved/<config_hash>.json
python -m eval.validate_golden --require-verified
python -m eval.harness --config configs/default.yaml --split fast   # results -> eval/results/<run_id>.json
python -m eval.noise_floor --config configs/default.yaml --split fast --runs 5  # add --existing-results a.json,b.json to resume
```

Not yet available (later phases — see `PROJECT_SPEC.md` §9 for the full target list):
`deploy/fetch_corpus.py`, `deploy/build_ingest_artifacts.py`, `deploy/record_demo_traces.py`
(all Phase 5), CI workflow wiring the harness as a gate (Phase 4 — see checklist above).

## Working agreements for this upgrade

- **Track the current phase's steps in this file as checkboxes; check them off as
  they land.** Once a phase closes, collapse its detail into "Facts carried forward"
  (only what later phases actually need) and remove the checklist — git history and
  the committed docs (`docs/NOISE_FLOOR.md` etc.) are the changelog, not this file.
- **Config-only ablations, no exceptions** (invariant 10) — if an ablation row requires
  editing code rather than a config value, config promotion is incomplete; fix that
  instead of hand-editing for one run.
- **Say when something is unverified, and verify for real when you can** — run it, don't
  just reason about it, whenever the environment allows (model downloads, live LLM
  calls, a real server launch, a fresh-clone check), per the spec's "provably works" goal.
- Never commit `api_keys.json`, `milvus.db`, `logs/`, `fail_logs.txt`, corpus text
  (`data/`, `artifacts/parsed_md/`), or any real key/secret.
- Prefer `git add <path>` over `git add -A`; check `git status` before staging.
- Message format: short imperative subject (`feat:`/`fix:`/`refactor:`/`docs:`/`chore:`), body explaining why when not obvious from the diff.
- **Commit messages describe the change, not the internal planning mechanics used to get
  there** — no "checkpoint N" language, no references to an in-session plan's own
  checkpoint numbering.
- Pause for user confirmation between phase milestones before proceeding to the next one.
