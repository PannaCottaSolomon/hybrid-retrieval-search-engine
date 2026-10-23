"""LibBOT Streamlit demo — owned by Member 6.

Run from the repo root:  streamlit run demo/app.py

Inputs available (all produced by the notebook):
  data/processed/corpus.parquet        doc_id, title, doc_text, sentences, ...
  data/processed/queries.parquet       query_id, claim, split
  data/processed/qrel_pairs.parquet    gold doc + evidence sentence ids per claim
  results/runs/<method>__test.parquet  query_id, doc_id, score, rank
  results/frozen_alphas.json           frozen fusion weights (Section 7)

TODO (Member 6): build the demo.
"""
