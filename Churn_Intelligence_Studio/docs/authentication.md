# Authentication and Credential Flow Review

## Finding

The original `50-Projects` repository was empty. There was no authentication code or documentation to trace. The project added here does not add sign-in or authorization either.

## Components

| Component | Authentication behavior |
|---|---|
| Streamlit UI (`app.py`) | No login, session identity, or role check |
| Dataset generation and model training (`src/data.py`, `src/train.py`) | Local files and local computation; no credentials |
| RAG retrieval (`src/assistant.py`) | TF-IDF retrieval against the included Markdown playbook; no credential |
| Ollama request (`src/assistant.py`) | HTTP POST to `http://localhost:11434/api/chat`; no `Authorization` header or bearer token |
| GitHub | GitHub account authentication is handled by GitHub and the connected GitHub integration; this application does not receive or manage that token |

## Request flow

```text
Browser -> local Streamlit app -> local model artifacts
Question -> TF-IDF retrieves playbook text -> app POSTs prompt to local Ollama -> answer shown in Streamlit
```

The model request body contains the selected Ollama model name, the user's question, and retrieved playbook text. The application sends it to a loopback address. It does not include an API key, cookie, or bearer token. There is no separate login request, token exchange, refresh flow, or protected endpoint in this codebase.

## Credential and token handling

- No application API keys, OAuth client secrets, passwords, JWTs, or session tokens are configured or stored.
- `OLLAMA_MODEL` selects a model name only; it is not a secret.
- Local ML and DL inference does not need network access or credentials.
- If the assistant is later pointed to a hosted provider, add a server-side credential source such as an environment variable or secret manager, send authorization only from the server, and keep secrets out of Git and browser code.
- Do not treat the current app as a multi-user or production-secure service.
