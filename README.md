# LibBOT: BM25–MiniLM hybrid retrieval for scientific evidence search

IS424 Data Mining · Group 5 (G2T5) · Hong Rong · Eileen · Adrian · Zu Kai · Pauline · Solomon

**Research question.** How do BM25, prefix MiniLM, windowed MiniLM, and their BM25-fused variants compare in ranking quality and retrieval efficiency on SciFact? Does MiniLM's 256-word-piece cutoff hide the expert-annotated evidence?

## Quick start

```bash
git clone https://github.com/yapzukai/libbot.git
cd libbot
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
nbstripout --install                                  # once per clone: strips notebook outputs on commit
jupyter lab LibBOT_pipeline.ipynb
```

Run **Section 1**, then either **Section 2** (downloads SciFact and rebuilds `data/processed/`) or **Checkpoint A** (loads the committed `data/processed/`), then your own section.

## Repo layout

```
libbot/
├── LibBOT_pipeline.ipynb     # the ONE shared notebook (sections owned per member)
├── requirements.txt
├── data/
│   ├── raw/                  # downloaded SciFact (git-ignored; Section 2 re-creates it)
│   └── processed/            # shared tables from Section 2 (committed)
├── cache/                    # embeddings / indexes (git-ignored; Sections 3–5)
├── results/
│   ├── runs/                 # <method>__<split>.parquet rankings (shared format)
│   ├── metrics/              # evaluation outputs, bootstrap CIs, diagnostic records
│   ├── efficiency/           # efficiency_log.jsonl
│   ├── figures/
│   └── config_snapshot.json  # config + environment of the last run
└── demo/app.py               # Streamlit demo (Member 6)
```

## Who owns what

| § | Section | Owner |
|---|---|---|
| 1 | Setup, config, shared helpers, data contract | Member 1 |
| 2 | Dataset loading + claim → document → evidence mapping | Member 2 |
| 3 | BM25 baseline | Member 3 |
| 4 | Prefix MiniLM | Member 4 |
| 5 | Windowed MiniLM | Member 5 |
| 6 | Truncation / evidence-retention diagnostic | Members 2 + 5 |
| 7 | Hybrid fusion + α tuning | Member 6 |
| 8 | Final evaluation (quality + efficiency) | Member 1 |
| 9 | Bootstrap CIs, results table, charts | Member 6 |
| 10 | Demo export, reproducibility log | Members 6, 1 |

## Data contract (summary; full version in notebook Section 1)

- All IDs are strings. `doc_text = (title + " " + text).strip()` is the one document string every method uses.
- **Rankings:** `results/runs/<method>__<split>.parquet` with columns `query_id, doc_id, score, rank`, top `RUN_DEPTH=1000` per query, raw scores (higher = better). Always write with `save_run()`, which validates the format.
- **Methods:** `bm25, prefix_minilm, window_minilm, hybrid_prefix, hybrid_window`. **Splits:** `train` (tuning), `test` (300 final queries).
- **Efficiency:** `log_efficiency(method, indexing_time_s=..., n_embeddings=..., storage_bytes=..., mean_latency_ms=..., p95_latency_ms=...)`.

## Important dataset note

BEIR SciFact's **test** qrels (300 claims) come from the original SciFact **dev** split, and BEIR **train** comes from original **train**. The original *test* split has no public labels. Section 2.3 checks this mapping from the data, and all evidence for the diagnostic is pulled from the matching original split.

## Git workflow (one shared notebook)

1. `git pull` before you start.
2. Edit only your section. Changes to Section 1 need a heads-up in the group chat.
3. `nbstripout` keeps outputs out of commits, so diffs stay small.
4. Commit with a clear message, e.g. `M3: BM25 runs for train/test`.
5. If two people need to edit at the same time, use branches (`m3-bm25`, `m4-prefix`, …) and merge one at a time through PRs.

## Fair evaluation protocol

Validate the rationale mapping (Section 2.7) → tune both hybrids on **train** with the same α grid → freeze → evaluate every configuration once on the 300 test queries → report wins, regressions, uncertainty and subgroup sizes. Test qrels and evidence locations are **never** used for tuning or window construction.
