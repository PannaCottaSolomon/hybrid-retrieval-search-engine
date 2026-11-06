"""Browse saved SciFact retrieval results.

Run from the repository root with: streamlit run demo/app.py
"""

from pathlib import Path
import json

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
METHODS = ["bm25", "prefix_minilm", "window_minilm", "hybrid_prefix", "hybrid_window"]


@st.cache_data(show_spinner=False)
def load_demo_data(root):
    root = Path(root)
    queries = pd.read_parquet(root / "data" / "processed" / "queries.parquet")
    corpus = pd.read_parquet(
        root / "data" / "processed" / "corpus.parquet",
        columns=["doc_id", "title", "doc_text"],
    )
    qrel_pairs = pd.read_parquet(root / "data" / "processed" / "qrel_pairs.parquet")
    evidence = pd.read_parquet(root / "data" / "processed" / "evidence_sentences.parquet")
    alpha_path = root / "results" / "frozen_alphas.json"
    alphas = json.loads(alpha_path.read_text(encoding="utf-8")) if alpha_path.exists() else {}
    return queries, corpus, qrel_pairs, evidence, alphas


@st.cache_data(show_spinner=False)
def load_test_run(root, method):
    if method not in METHODS:
        raise ValueError(f"Unknown retrieval method: {method}")
    path = Path(root) / "results" / "runs" / f"{method}__test.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing test run: {path}")
    run = pd.read_parquet(path, columns=["query_id", "doc_id", "score", "rank"])
    run["query_id"] = run["query_id"].astype(str)
    run["doc_id"] = run["doc_id"].astype(str)
    return run


st.set_page_config(page_title="SciFact evidence retrieval", layout="wide")
st.title("SciFact evidence retrieval")
st.caption("Explore saved test-set rankings. Hybrid weights were selected on the train split.")

queries, corpus, qrel_pairs, evidence, alphas = load_demo_data(str(ROOT))
queries = queries.loc[queries["split"] == "test"].copy()
queries["query_id"] = queries["query_id"].astype(str)
qrel_pairs = qrel_pairs.loc[qrel_pairs["split"] == "test"].copy()
qrel_pairs["query_id"] = qrel_pairs["query_id"].astype(str)
qrel_pairs["doc_id"] = qrel_pairs["doc_id"].astype(str)
evidence = evidence.loc[evidence["split"] == "test"].copy()
evidence["query_id"] = evidence["query_id"].astype(str)
evidence["doc_id"] = evidence["doc_id"].astype(str)
corpus["doc_id"] = corpus["doc_id"].astype(str)

with st.sidebar:
    st.header("Search settings")
    method = st.selectbox("Ranking method", METHODS, format_func=lambda value: value.replace("_", " ").title())
    query_filter = st.text_input("Filter claims")
    top_k = st.slider("Documents to show", min_value=5, max_value=20, value=10)

filtered_queries = queries
if query_filter.strip():
    filtered_queries = queries[queries["claim"].str.contains(query_filter.strip(), case=False, na=False)]
if filtered_queries.empty:
    st.info("No test claims match that filter.")
    st.stop()

query_lookup = filtered_queries.set_index("query_id")["claim"].to_dict()
query_id = st.selectbox(
    "Claim",
    options=filtered_queries["query_id"].tolist(),
    format_func=lambda value: f"{value}: {query_lookup[value]}",
)
claim = query_lookup[query_id]
st.markdown(f"**Claim:** {claim}")

run = load_test_run(str(ROOT), method)
ranked = run.loc[run["query_id"] == query_id].sort_values("rank").head(top_k)
ranked = ranked.merge(corpus, on="doc_id", how="left", validate="many_to_one")
pair_info = qrel_pairs.loc[
    qrel_pairs["query_id"] == query_id,
    ["doc_id", "relevance", "label", "has_evidence"],
]
ranked = ranked.merge(pair_info, on="doc_id", how="left", validate="one_to_one")
ranked["relevant"] = ranked["relevance"].notna()
ranked["has_evidence"] = ranked["has_evidence"].fillna(False).astype(bool)

if method in alphas:
    st.caption(f"Frozen {method} alpha: {float(alphas[method]):.2f}")

st.subheader(f"Top {len(ranked)} documents")
display_columns = ["rank", "score", "title", "doc_id", "relevant", "label", "has_evidence"]
st.dataframe(ranked[display_columns], hide_index=True, use_container_width=True)

document_options = ranked["doc_id"].tolist()
if document_options:
    doc_lookup = ranked.set_index("doc_id")["title"].fillna("").to_dict()
    selected_doc = st.selectbox(
        "Document details",
        options=document_options,
        format_func=lambda value: f"{doc_lookup[value]} ({value})",
    )
    selected = ranked.loc[ranked["doc_id"] == selected_doc].iloc[0]
    st.subheader(str(selected["title"] or selected_doc))
    st.write(selected["doc_text"])

    doc_evidence = evidence.loc[
        (evidence["query_id"] == query_id) & (evidence["doc_id"] == selected_doc)
    ].sort_values(["rationale_set", "sentence_idx"])
    if doc_evidence.empty:
        st.caption("No expert evidence sentence is annotated for this query/document pair.")
    else:
        st.markdown("**Annotated evidence sentences**")
        for row in doc_evidence.itertuples(index=False):
            st.markdown(f"- {row.sentence_text}")
else:
    st.info("This method has no saved results for the selected claim.")
