# app.py
# Streamlit Sentiment App: Naive Bayes, ANN, DistilBERT (with sliding window)
# ---------------------------------------------------------------
import os, io, time, re, json, math, joblib, html, unicodedata
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from huggingface_hub import hf_hub_download

# Optional heavy imports guarded to speed up initial load
from typing import List, Dict, Tuple

# -------------------------
# Config & paths
# -------------------------
st.set_page_config(page_title="Sentiment Analysis Suite", layout="wide")
# MODELS_DIR = "models/"
NB_PATH = "noobrs/nb-movie-sentiment"
NB_FILE = "nb-sentiment.joblib"

ANN_PATH = "noobrs/ann-movie-sentiment"
ANN_FILE = "imdb_mlp_tfidf.keras"
ANN_TFIDF_FILE = "tfidf.pkl"

DISTILBERT_DIR = "apple-pie-vs/distilbert-movie-sentiment" # huggingface hub model ID

MAX_LEN = 512  # DistilBERT max length
STRIDE = 256   # DistilBERT sliding window stride

# -------------------------
# Utilities
# -------------------------
POS_LABELS = {"positive", "pos", "1", 1}

def clean_text(x):
    if not isinstance(x, str):
        return ""
    # Unescape & strip HTML
    x = html.unescape(x)
    x = BeautifulSoup(x, "lxml").get_text(separator=" ")

    # Unicode normalize + unify curly quotes to straight ones
    x = unicodedata.normalize("NFKC", x)
    x = x.replace("“", "'").replace("”", "'").replace("‘", "'").replace("’", "'").replace('"', "'")

    # Neutralize obvious artifacts
    x = re.sub(r"(https?://\S+)|(\w+\.\w+/\S+)", " ", x)
    x = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", x)

    # runs of 2+ asterisks → single *
    x = re.sub(r"\{2,}", "", x)

    # collapse any run of -, – or — to a single em dash, with spacing
    x = re.sub(r"\s*[-–—]{2,}\s*", " — ", x)

    # "" → "   and   '' → '
    x = re.sub(r'([\'\"])\1+', r'\1', x)  # collapse immediate repeats
    # also clean cases with whitespace between repeated quotes: "  " → "
    x = re.sub(r'([\'"])\s+\1', r'\1', x)

    # cap !!!!! or ????? at two; dots at an ellipsis
    x = re.sub(r"([!?])\1{2,}", r"\1\1", x)   # keep at most two
    x = re.sub(r"\.{3,}", "…", x)

    # 5) Remove control chars & collapse whitespace
    x = re.sub(r"[\u0000-\u001F\u007F]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def df_to_download(df: pd.DataFrame, filename: str) -> Tuple[bytes, str]:
    csv = df.to_csv(index=False).encode("utf-8")
    return csv, filename

# -------------------------
# Model wrappers
# -------------------------
@st.cache_resource(show_spinner=True)
def load_nb():
    try:
        nb_path = hf_hub_download(NB_PATH, filename=NB_FILE)
        return joblib.load(nb_path)
    except Exception as e:
        return None

@st.cache_resource(show_spinner=True)
def load_ann():
    try:
        tfidf_p = hf_hub_download(ANN_PATH, filename=ANN_TFIDF_FILE)
        ann_p   = hf_hub_download(ANN_PATH, filename=ANN_FILE)
        import tensorflow as tf, joblib
        return {"model": tf.keras.models.load_model(ann_p),
                "tfidf": joblib.load(tfidf_p)}
    except Exception as e:
        return None

@st.cache_resource(show_spinner=True)
def load_distilbert(model_dir=DISTILBERT_DIR):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    path = model_dir
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModelForSequenceClassification.from_pretrained(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    mdl.eval()
    return tok, mdl, device

def _prob_pos_from_sklearn_pipeline(pipe, texts: List[str]) -> np.ndarray:
    """
    Returns probability of positive class for a sklearn classifier pipeline
    Assumes binary classification with classes_ aligned to [0, 1] or containing a positive-like label.
    """
    if not hasattr(pipe, "predict_proba"):
        # If final estimator doesn't support proba, fall back to decision_function -> sigmoid
        if hasattr(pipe, "decision_function"):
            from scipy.special import expit
            scores = pipe.decision_function(texts)
            # Map to 0..1, assume positive on high score
            probs = expit(scores)
            return np.array(probs).reshape(-1)
        else:
            # last resort: predict and map to 0/1
            preds = pipe.predict(texts)
            return np.array([1.0 if str(p).lower() in POS_LABELS else 0.0 for p in preds])

    probs = pipe.predict_proba(texts)
    # Locate index of "positive-like" class if possible
    classes = [str(c).lower() for c in getattr(pipe, "classes_", [0, 1])]
    if "positive" in classes:
        idx = classes.index("positive")
    elif "pos" in classes:
        idx = classes.index("pos")
    elif "1" in classes:
        idx = classes.index("1")
    else:
        # assume class order [neg, pos]
        idx = -1
    return probs[:, idx]

def _prob_pos_from_ann(ann_dict, texts: List[str]) -> np.ndarray:
    """
    Returns probability of positive class for ANN model with TF-IDF
    """
    import tensorflow as tf
    model = ann_dict["model"]
    tfidf = ann_dict["tfidf"]
    
    # Transform texts using TF-IDF
    X = tfidf.transform(texts)
    
    # Predict with the neural network
    probs = model.predict(X.toarray(), verbose=0)
    
    # If binary classification, return probability of positive class
    if probs.shape[1] == 1:
        return probs.flatten()  # sigmoid output
    else:
        return probs[:, 1]  # softmax output, class 1 is positive

def _distilbert_probs_sliding(texts: List[str], max_len=512, stride=256) -> np.ndarray:
    import torch, inspect, numpy as np
    tok, mdl, device = load_distilbert()

    # work out the positive-class index safely
    id2label = getattr(mdl.config, "id2label", None)
    num_labels = getattr(mdl.config, "num_labels", 2)
    pos_idx = 1
    if isinstance(id2label, dict) and id2label:
        label_map = {int(k): str(v).lower() for k, v in id2label.items()}
        for k, v in label_map.items():
            if "pos" in v:
                pos_idx = k
                break

    # only pass args that forward() accepts (avoid overflow_to_sample_mapping, etc.)
    allowed_forward_args = set(inspect.signature(mdl.forward).parameters.keys())

    pos_probs = []
    for t in texts:
        if not t or not t.strip():
            pos_probs.append(0.5)
            continue

        enc = tok(
            t,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            stride=stride,
            return_overflowing_tokens=True,
        )

        num_chunks = enc["input_ids"].shape[0]
        probs_each = []
        with torch.no_grad():
            for i in range(num_chunks):
                # filter to keys the model actually supports
                inputs = {
                    k: v[i:i+1].to(device)
                    for k, v in enc.items()
                    if isinstance(v, torch.Tensor) and k in allowed_forward_args
                }
                out = mdl(**inputs)
                logits = out.logits  # shape (1, C)
                if num_labels == 1:
                    p = torch.sigmoid(logits).item()
                else:
                    p = torch.softmax(logits, dim=-1)[0, pos_idx].item()
                probs_each.append(p)

        pos_probs.append(float(np.mean(probs_each)))

    return np.array(pos_probs)

# High-level prediction orchestrator
def run_models(texts: List[str],
               use_nb: bool,
               use_ann: bool,
               use_distilbert: bool) -> pd.DataFrame:
    # Preprocess
    cleaned = [clean_text(t) for t in texts]

    results = pd.DataFrame({"text": texts, "cleaned": cleaned})

    if use_nb:
        nb = load_nb()
        if nb is None:
            results["nb_prob_pos"] = np.nan
            results["nb_label"] = "MODEL NOT FOUND"
        else:
            probs = _prob_pos_from_sklearn_pipeline(nb, cleaned)
            results["nb_prob_pos"] = probs
            results["nb_label"] = np.where(probs >= 0.5, "positive", "negative")

    if use_ann:
        ann = load_ann()
        if ann is None:
            results["ann_prob_pos"] = np.nan
            results["ann_label"] = "MODEL NOT FOUND"
        else:
            probs = _prob_pos_from_ann(ann, cleaned)
            results["ann_prob_pos"] = probs
            results["ann_label"] = np.where(probs >= 0.5, "positive", "negative")

    if use_distilbert:
        probs = _distilbert_probs_sliding(cleaned,
                                          max_len=MAX_LEN,
                                          stride=STRIDE)
        results["distilbert_prob_pos"] = probs
        results["distilbert_label"] = np.where(probs >= 0.5, "positive", "negative")

    return results

# -------------------------
# Charts
# -------------------------
def distribution_charts(df: pd.DataFrame, model_key: str, title_prefix: str):
    # Count pos/neg
    counts = df[model_key].value_counts(dropna=False).rename_axis("label").reset_index(name="count")
    st.subheader(f"{title_prefix}: Polarity Distribution ({model_key})")
    left, right = st.columns(2)
    with left:
        pie = alt.Chart(counts).mark_arc().encode(theta="count:Q", color="label:N", tooltip=["label", "count"])
        st.altair_chart(pie, use_container_width=True)
    with right:
        bar = alt.Chart(counts).mark_bar().encode(x="label:N", y="count:Q", tooltip=["label", "count"])
        st.altair_chart(bar, use_container_width=True)

def multi_model_comparison_chart(df: pd.DataFrame, selected_models: List[str], title: str):
    """
    Create comparison chart with neg/pos as x-axis, result count as y-axis, and model type as legend
    """
    # Prepare data for all models
    comparison_data = []
    model_names = {"nb": "Naive Bayes", "ann": "ANN", "distilbert": "DistilBERT"}
    
    for model in selected_models:
        label_col = f"{model}_label"
        if label_col in df.columns:
            counts = df[label_col].value_counts(dropna=False)
            for sentiment, count in counts.items():
                comparison_data.append({
                    "sentiment": sentiment,
                    "count": count,
                    "model": model_names.get(model, model.upper())
                })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.subheader(title)
        
        # Create grouped bar chart with sentiment on x-axis and models grouped side by side
        chart = alt.Chart(comparison_df).mark_bar().encode(
            x=alt.X("sentiment:N", title="Sentiment", sort=["negative", "positive"]),
            y=alt.Y("count:Q", title="Count"),
            color=alt.Color("model:N", title="Model", 
                          scale=alt.Scale(range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])),
            xOffset=alt.XOffset("model:N"),
            tooltip=["model:N", "sentiment:N", "count:Q"]
        ).properties(
            width=400,
            height=300
        )
        
        st.altair_chart(chart, use_container_width=True)

def comparison_chart(df: pd.DataFrame, prob_cols: List[str], title: str):
    # Average positive probability by model
    melted = df.melt(id_vars=["text"], value_vars=prob_cols, var_name="model", value_name="prob_pos").dropna()
    # Tidy model names
    name_map = {c: c.replace("_prob_pos", "").upper() for c in prob_cols}
    melted["model"] = melted["model"].map(name_map)
    st.subheader(title)
    bar = alt.Chart(melted).mark_bar().encode(
        x=alt.X("model:N", title="Model"),
        y=alt.Y("mean(prob_pos):Q", title="Mean P(positive)"),
        tooltip=[alt.Tooltip("mean(prob_pos):Q", title="Mean P(pos)", format=".4f")]
    )
    st.altair_chart(bar, use_container_width=True)

def single_text_stacked_chart(row: pd.Series):
    """
    Show a stacked bar chart for single inference with all models
    """
    # Prepare data for stacked chart
    chart_data = []
    model_names = {"nb": "Naive Bayes", "ann": "ANN", "distilbert": "DistilBERT"}
    
    for m in ["nb", "ann", "distilbert"]:
        prob_col = f"{m}_prob_pos"
        if prob_col in row and not pd.isna(row[prob_col]):
            prob_pos = float(row[prob_col])
            prob_neg = 1.0 - prob_pos
            
            # Add positive probability
            chart_data.append({
                "model": model_names[m],
                "sentiment": "Positive",
                "probability": prob_pos,
                "order": 1
            })
            
            # Add negative probability
            chart_data.append({
                "model": model_names[m], 
                "sentiment": "Negative",
                "probability": prob_neg,
                "order": 0
            })
    
    if not chart_data:
        return
        
    chart_df = pd.DataFrame(chart_data)
    st.subheader("Model Confidence Breakdown")
    
    # Create stacked bar chart
    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("model:N", title="Model"),
        y=alt.Y("probability:Q", title="Probability", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("sentiment:N", title="Sentiment",
                       scale=alt.Scale(domain=["Negative", "Positive"], 
                                     range=["#ff4444", "#44aa44"])),
        order=alt.Order("order:O"),
        tooltip=["model:N", "sentiment:N", alt.Tooltip("probability:Q", format=".4f")]
    ).properties(
        height=300
    )
    
    st.altair_chart(chart, use_container_width=True)

def single_text_chart(row: pd.Series):
    # Show a neat horizontal bar of probabilities
    pairs = []
    for m in ["nb", "ann", "distilbert"]:
        col = f"{m}_prob_pos"
        if col in row and not pd.isna(row[col]):
            pairs.append({"model": m.upper(), "P(positive)": float(row[col])})
    if not pairs:
        return
    df = pd.DataFrame(pairs)
    st.subheader("Confidence by Model")
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("P(positive):Q", scale=alt.Scale(domain=[0,1])),
        y=alt.Y("model:N", sort="-x"),
        tooltip=[alt.Tooltip("P(positive):Q", format=".4f"), "model:N"]
    )
    st.altair_chart(chart, use_container_width=True)

# -------------------------
# UI
# -------------------------
st.title("Movie Review Sentiment Analysis")
st.caption("Naive Bayes • ANN • DistilBERT — Unified Smart Inference")

# -------------------------
# Smart Inference
# -------------------------
st.subheader("Smart Inference")
st.markdown("**Input your data in any of these ways:**")

texts = []
dataset_name = "batch"

# Combined input area
col1, col2 = st.columns([3, 1])

with col1:
    # File upload
    up = st.file_uploader("Upload CSV file (optional)", type=["csv"])
    
    # Text input area
    multi = st.text_area(
        "Or paste/type reviews here",
        height=200,
        placeholder="Separate multiple reviews with new lines",
        help="You can enter a single review, multiple reviews (one per line), or upload a CSV file above."
    )

with col2:
    st.markdown("**Input detected:**")
    input_type = st.empty()
    input_count = st.empty()

# Smart input detection and processing
if up:
    try:
        df_up = pd.read_csv(up)
        col = st.selectbox("Which column contains the reviews?", list(df_up.columns))
        texts = df_up[col].astype(str).tolist()
        dataset_name = os.path.splitext(up.name)[0]
        input_type.success("CSV Upload")
        input_count.info(f"{len(texts)} reviews")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

elif multi.strip():
    lines = [line.strip() for line in multi.splitlines() if line.strip()]
    if len(lines) == 1:
        # Single review (possibly multiline)
        texts = [multi.strip()]
        dataset_name = "single_review"
        input_type.success("Single Review")
        input_count.info("1 review")
    else:
        # Multiple reviews (one per line)
        texts = lines
        dataset_name = "multiline_input"
        input_type.success("Multiple Reviews")
        input_count.info(f"{len(texts)} reviews")

# Model selection for inference (always visible)
st.markdown("---")
st.markdown("**Select Model for Inference:**")


model_choice = st.radio(
    "Choose one:",
    ["All Models", "Naive Bayes", "Artificial Neural Network (ANN)", "DistilBERT"],
    index=0,
    help="Select either one specific model or all models for comparison"
)

# Set model flags based on choice
if model_choice == "All Models":
    use_nb_infer = use_ann_infer = use_distilbert_infer = True
elif model_choice == "Naive Bayes":
    use_nb_infer, use_ann_infer, use_distilbert_infer = True, False, False
elif model_choice == "Artificial Neural Network (ANN)":
    use_nb_infer, use_ann_infer, use_distilbert_infer = False, True, False
elif model_choice == "DistilBERT":
    use_nb_infer, use_ann_infer, use_distilbert_infer = False, False, True

st.markdown("---")

# Run inference button (always visible)
if st.button("Run", type="primary", disabled=not texts):
    if not texts:
        st.warning("Please provide input text or upload a CSV file first.")
    else:
        with st.spinner("Running models…"):
            df = run_models(
                texts,
                use_nb=use_nb_infer, use_ann=use_ann_infer, use_distilbert=use_distilbert_infer
            )
        st.success("Done.")

        # Determine output format based on input count and model selection
        num_texts = len(texts)
        num_models_selected = sum([use_nb_infer, use_ann_infer, use_distilbert_infer])
        is_single_text = num_texts == 1
        is_single_model = num_models_selected == 1
        is_all_models = model_choice == "All Models"

        # Prepare columns for display
        show_cols = ["text"]
        selected_models = []
        for m in ["nb", "ann", "distilbert"]:
            if f"{m}_label" in df.columns:
                show_cols += [f"{m}_label", f"{m}_prob_pos"]
                selected_models.append(m)

        # Case 1: Single text + Single model - Show sentiment result only
        if is_single_text and is_single_model:
            st.markdown("### Sentiment Result")
            row = df.iloc[0]
            
            # Find the active model
            for m in selected_models:
                label_col = f"{m}_label"
                prob_col = f"{m}_prob_pos"
                model_name = {"nb": "Naive Bayes", "ann": "Artificial Neural Network (ANN)", "distilbert": "DistilBERT"}[m]

                # Create a prominent result display
                sentiment = row[label_col]
                confidence = row[prob_col]
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if sentiment == "positive":
                        st.success(f"**{sentiment.upper()}**")
                    else:
                        st.error(f"**{sentiment.upper()}**")
                    
                    st.metric(
                        label=f"{model_name} Confidence",
                        value=f"{confidence:.4f}",
                        delta=f"P(positive) = {confidence:.4f}"
                    )


        # Case 2: Single text + All models - Show all three results
        elif is_single_text and is_all_models:
            st.markdown("### Model Comparison")
            row = df.iloc[0]
            
            cols = st.columns(len(selected_models))
            model_names = {"nb": "Naive Bayes", "ann": "Artificial Neural Network (ANN)", "distilbert": "DistilBERT"}

            for i, m in enumerate(selected_models):
                label_col = f"{m}_label"
                prob_col = f"{m}_prob_pos"
                sentiment = row[label_col]
                confidence = row[prob_col]
                
                with cols[i]:
                    st.markdown(f"**{model_names[m]}**")
                    if sentiment == "positive":
                        st.success(f"{sentiment.upper()}")
                    else:
                        st.error(f"{sentiment.upper()}")
                    st.metric("Confidence", f"{confidence:.4f}")
            
            # Show comparison chart
            single_text_chart(row)

        # Case 3: Multiple texts + Single model - Show pie chart + scrollable grid
        elif not is_single_text and is_single_model:
            # Find the active model
            active_model = selected_models[0]
            label_col = f"{active_model}_label"
            model_name = {"nb": "Naive Bayes", "ann": "Artificial Neural Network (ANN)", "distilbert": "DistilBERT"}[active_model]
            
            st.markdown(f"### {model_name} Results ({num_texts} texts)")
            
            # Pie chart for sentiment distribution
            distribution_charts(df, label_col, model_name)
            
            # Scrollable grid of results
            st.markdown("### Detailed Results")
            display_cols = ["text", label_col, f"{active_model}_prob_pos"]
            st.dataframe(
                df[display_cols].rename(columns={
                    label_col: "Sentiment",
                    f"{active_model}_prob_pos": "Confidence"
                }), 
                use_container_width=True, 
                height=400
            )

        # Case 4: Multiple texts + All models - Show individual charts + comparison + scrollable grid
        else:  # Multiple texts + All models
            st.markdown(f"### Multi-Model Analysis ({num_texts} texts)")
            
            # Individual distribution charts for each model (same as single model case)
            for m in selected_models:
                if f"{m}_label" in df.columns:
                    model_name = {"nb": "Naive Bayes", "ann": "Artificial Neural Network (ANN)", "distilbert": "DistilBERT"}[m]
                    distribution_charts(df, f"{m}_label", model_name)
            
            # Multi-model comparison chart with neg/pos as x-axis and models as legend
            multi_model_comparison_chart(df, selected_models, "All Models Comparison by Sentiment")
            
            # Scrollable grid with all model results
            st.markdown("### Detailed Results (All Models)")
            st.dataframe(df[show_cols], use_container_width=True, height=400)

        # Export functionality (always available)
        st.markdown("---")
        ts = time.strftime("%Y%m%d-%H%M%S")
        outname = f"{dataset_name}_sentiment_{ts}.csv"
        csv_bytes, fname = df_to_download(df[show_cols], outname)
        st.download_button("Export results to CSV", data=csv_bytes, file_name=fname, mime="text/csv")
