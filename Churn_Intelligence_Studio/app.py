"""Streamlit interface for churn model comparisons and local RAG."""

from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st
import torch

from src.assistant import ask_local_model, retrieve
from src.features import FEATURES
from src.train import ChurnMLP

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(page_title="Churn Intelligence Studio", page_icon="??", layout="wide")
st.title("Customer Churn Intelligence Studio")
st.caption("Learn how classical ML, deep learning, and a local GenAI assistant fit into one workflow.")

with st.sidebar:
    st.header("Customer profile")
    plan = st.selectbox("Plan", ["Basic", "Plus", "Premium"])
    contract = st.selectbox("Contract", ["Monthly", "Annual", "Two-year"])
    tenure = st.slider("Tenure (months)", 1, 72, 12)
    charge = st.slider("Monthly charge", 20, 180, 80)
    support = st.slider("Support calls in last 90 days", 0, 8, 1)
    late = st.slider("Late payments in last 90 days", 0, 5, 0)
    usage = st.slider("Monthly usage hours", 1, 100, 30)

record = pd.DataFrame(
    [[plan, contract, tenure, charge, support, late, usage]], columns=FEATURES
)

if not (ARTIFACTS / "metrics.json").exists():
    st.info("Train the models first: run `python -m src.data`, then `python -m src.train`.")
else:
    metrics = pd.read_json(ARTIFACTS / "metrics.json", typ="series").to_dict()
    rows = []
    for name, title in [
        ("logistic_regression", "Logistic Regression"),
        ("random_forest", "Random Forest"),
    ]:
        model = joblib.load(ARTIFACTS / f"{name}.joblib")
        rows.append({"Model": title, "Churn probability": float(model.predict_proba(record)[0, 1])})

    mlp_bundle = torch.load(ARTIFACTS / "pytorch_mlp.pt", map_location="cpu", weights_only=True)
    preprocessor = joblib.load(ARTIFACTS / "pytorch_preprocessor.joblib")
    mlp = ChurnMLP(mlp_bundle["input_size"])
    mlp.load_state_dict(mlp_bundle["state_dict"])
    mlp.eval()
    transformed = preprocessor.transform(record).astype("float32")
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    with torch.no_grad():
        probability = torch.sigmoid(mlp(torch.tensor(transformed))).item()
    rows.append({"Model": "PyTorch MLP", "Churn probability": probability})

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Model comparison")
        result_df = pd.DataFrame(rows)
        st.dataframe(result_df.style.format({"Churn probability": "{:.1%}"}), hide_index=True)
        st.bar_chart(result_df.set_index("Model"))
        st.caption("These are predictions from synthetic demo data, not real customer risk assessments.")
    with right:
        st.subheader("Held-out evaluation")
        st.json(metrics)

st.divider()
st.subheader("Retention assistant (local GenAI + RAG)")
st.write("The app retrieves related notes from the included playbook and sends them to Ollama running on this computer.")
question = st.text_input("Ask about a respectful next step", placeholder="What should I review after repeated support calls?")
if st.button("Ask assistant", disabled=not question.strip()):
    evidence = retrieve(question)
    with st.expander("Retrieved playbook notes", expanded=False):
        st.write("\n\n---\n\n".join(evidence) if evidence else "No close match found.")
    try:
        st.markdown(ask_local_model(question, evidence))
    except requests.RequestException as exc:
        st.error(f"Could not reach local Ollama. Start Ollama and pull the configured model. Details: {exc}")
