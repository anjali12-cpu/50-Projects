# Customer Churn Intelligence Studio

An educational end-to-end project that combines classical machine learning, a PyTorch deep-learning model, and a local retrieval-augmented GenAI assistant. It uses generated customer data so the project runs without downloading private or licensed data.

## What it does

1. Generates a reproducible customer churn dataset.
2. Trains and evaluates Logistic Regression and Random Forest baselines.
3. Trains and evaluates a small PyTorch multilayer perceptron (MLP).
4. Serves churn predictions and a grounded retention assistant in Streamlit.

The generated data is synthetic and intended for learning. Model scores are not evidence of real-world performance.

## Project layout

```text
Churn_Intelligence_Studio/
��� app.py
��� requirements.txt
��� data/                         # generated locally; ignored by Git
��� artifacts/                    # trained locally; ignored by Git
��� docs/retention_playbook.md
��� src/
    ��� data.py
    ��� features.py
    ��� train.py
    ��� assistant.py
```

## Run it

### 1. Create an environment and install packages

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Generate data and train models

```bash
python -m src.data
python -m src.train
```

Training writes model files and a metrics report to `artifacts/`.

### 3. Start the app

```bash
streamlit run app.py
```

Choose customer attributes in the sidebar to compare model predictions. The assistant retrieves relevant passages from `docs/retention_playbook.md` and asks a local Ollama model to draft an explanation.

## Optional local GenAI setup

Install [Ollama](https://ollama.com), start its local service, and pull a model (for example `ollama pull llama3.2`). Set `OLLAMA_MODEL` if you want a different installed model. The app calls Ollama at `http://localhost:11434`; it does not require an OpenAI API key or any cloud token. If Ollama is unavailable, the app shows a clear setup message and the ML/DL predictions still work.

## Model and data flow

```text
Synthetic customer rows ��> validation / preprocessing ��> train/test split
                                                      ���> Logistic Regression
                                                      ���> Random Forest
                                                      ���> PyTorch MLP
Streamlit form ��> all three saved models ��> churn probabilities
User question ��> TF-IDF playbook retrieval ��> local Ollama ��> grounded answer
```

Categorical fields are one-hot encoded and numeric fields are standardized using preprocessing fitted only on the training split. The test split is held out for evaluation. The assistant uses local document retrieval and a local model endpoint; do not put secrets or personal customer data into this demo.

## Authentication and credentials

The repository started empty and contains no authentication implementation, login component, protected route, or token flow. This demo also has no user accounts or sign-in. Model training and prediction need no credentials. The optional GenAI feature talks to a local Ollama service and does not send a bearer token. No API keys or tokens are stored in the code. If you adapt the app to a hosted model, load its credential from an environment variable or secret manager; never commit it.

## Limitations

- Generated labels follow a simple simulated pattern and can make the task easier than real churn prediction.
- Deep learning is included as a teaching baseline; tree ensembles often perform better on small tabular datasets.
- No authentication, authorization, database, monitoring, or production privacy controls are included.
- Evaluate calibration, fairness, leakage, and business impact with a properly sourced dataset before real use.

## Troubleshooting

- `FileNotFoundError` for models: run `python -m src.data` and `python -m src.train` first.
- Torch installation: use the installation command recommended by the official PyTorch site if the default package does not match your CPU/GPU environment.
- GenAI not available: start Ollama locally and confirm the selected model has been pulled. The rest of the app works without it.
