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

- **Multiple model endpoints** — assistant, reason, fast, research, compound, agent, and teacher, each mapped to a dedicated model
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
| POST   | `/api/agent`    | AI agent that mimics Shubham Kaushik  |
| POST   | `/api/teacher`  | AI teacher for DocNest — answers AI/ML questions with history |
| GET    | `/docs`         | Interactive Swagger UI (auto-generated) |

### Model routing

| Endpoint        | Model                   |
| --------------- | ----------------------- |
| `/api/assistant`| `openai/gpt-oss-20b`    |
| `/api/reason`   | `openai/gpt-oss-120b`   |
| `/api/fast`     | `openai/gpt-oss-20b`    |
| `/api/research` | `groq/compound-mini`    |
| `/api/compound` | `groq/compound`         |
| `/api/agent`    | `openai/gpt-oss-120b`   |
| `/api/teacher`  | `openai/gpt-oss-120b`   |

The teacher endpoint accepts an optional `history` array of `{role, content}`
turns (limited to the last 20) so the chat can carry multi-turn context:

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

### Teacher chat example

```bash
curl -X POST http://localhost:8000/api/teacher \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How does attention work?",
    "history": [
      {"role": "user", "content": "What is a transformer?"},
      {"role": "assistant", "content": "A transformer is a neural network architecture..."}
    ]
  }'
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
