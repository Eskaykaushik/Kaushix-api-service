---
title: Kaushix API
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
app_port: 7860
pinned: false
---

# Kaushix API

Shared FastAPI backend for Kaushix Labs and static websites. It exposes a simple JSON API backed by multiple LLM models served through the Groq platform.

## Features

- **Multiple model endpoints** — assistant, reason, fast, research, and compound, each mapped to a dedicated model
- **FastAPI** — modern, async-ready Python framework with automatic OpenAPI docs at `/docs`
- **CORS enabled** — ready to be called directly from any browser-based frontend
- **Zero UI dependencies** — pure JSON API, no Gradio runtime required

## Endpoints

| Method | Path            | Purpose                                |
| ------ | --------------- | -------------------------------------- |
| GET    | `/`             | Service metadata                       |
| GET    | `/api/health`   | Health check                           |
| POST   | `/api/assistant`| General assistant responses            |
| POST   | `/api/reason`   | Deep reasoning tasks                   |
| POST   | `/api/fast`     | Quick, low-latency responses           |
| POST   | `/api/research` | Research-oriented responses            |
| POST   | `/api/compound` | Compound / multi-step analysis         |
| GET    | `/docs`         | Interactive Swagger UI (auto-generated) |

### Model routing

| Endpoint        | Model                   |
| --------------- | ----------------------- |
| `/api/assistant`| `openai/gpt-oss-20b`    |
| `/api/reason`   | `openai/gpt-oss-120b`   |
| `/api/fast`     | `openai/gpt-oss-20b`    |
| `/api/research` | `groq/compound-mini`    |
| `/api/compound` | `groq/compound`         |

## Quickstart

### Prerequisites

- Python 3.12+
- A Groq API key from [console.groq.com](https://console.groq.com)

### Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set your API key:

```bash
export GROQ_API_KEY="your-key-here"
```

### Run

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`.

### Example request

```bash
curl -X POST http://localhost:8000/api/assistant \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 2 + 2?"}'
```

Response:

```json
{
  "response": "4"
}
```

## Configuration

| Variable        | Required | Description       |
| --------------- | -------- | ----------------- |
| `GROQ_API_KEY`  | Yes      | Groq API key      |

The model mapping is defined in the `MODELS` dictionary in `app.py`.

## Testing

```bash
pytest tests
```

## License

Proprietary. All rights reserved. © Kaushix Labs.
