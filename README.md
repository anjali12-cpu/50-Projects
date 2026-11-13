# Customer Churn Intelligence Studio

An educational end-to-end project combining classical machine learning, a PyTorch deep-learning model, and a local retrieval-augmented GenAI assistant. The reproducible synthetic data generator avoids external datasets.

## Workflow

1. Generate customer churn data: `python -m src.data`
2. Train/evaluate Logistic Regression, Random Forest, and a PyTorch MLP: `python -m src.train`
3. Launch the Streamlit app: `streamlit run app.py`
4. Compare churn probabilities and ask the playbook assistant a question.

The app evaluates models on a held-out test split. Categorical data is one-hot encoded and numeric data is standardized using training-only fitted preprocessing. Synthetic model scores are for learning and do not represent real-world performance.

## Setup

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.data
python -m src.train
streamlit run app.py
```

## GenAI

The RAG assistant retrieves relevant sections from `docs/retention_playbook.md` using TF-IDF and sends those notes to a local Ollama model at `http://localhost:11434`. Install Ollama and pull a model such as `llama3.2`. The prediction app works without Ollama. No OpenAI key, cloud token, API key, or bearer token is required or stored.

## Authentication and credential handling

The repository was empty before this project; it had no authentication components, protected routes, or token flow to explain. This project has no user accounts or sign-in. Local model training and inference need no credentials. The optional GenAI path uses a local Ollama endpoint without sending a bearer token. If adapted to a hosted model, store its credential in an environment variable or secret manager and do not commit it.

## Files

- `app.py`: Streamlit prediction and assistant interface.
- `src/data.py`: reproducible synthetic dataset generator.
- `src/features.py`: shared feature definitions and preprocessing.
- `src/train.py`: classical ML and PyTorch MLP training/evaluation.
- `src/assistant.py`: local document retrieval and Ollama request.
- `docs/retention_playbook.md`: grounded example knowledge base.

## Limitations

Generated labels follow a simulation rule and may make prediction easier than real churn analysis. Deep learning is included as a teaching baseline; tree models often work better on small tabular data. This project has no production privacy controls, monitoring, database, authentication, or authorization. Validate quality, fairness, leakage, and business impact before real use.
