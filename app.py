import os

import gradio as gr
from fastapi import FastAPI, HTTPException
from groq import Groq
from pydantic import BaseModel


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

MODEL = "llama-3.3-70b-versatile"
SYSTEM_PROMPT = "You are a helpful and concise AI assistant."


# ─────────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────────

app = FastAPI(
    title="Kaushix API",
    version="0.1.0",
)


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

class AssistantRequest(BaseModel):
    message: str


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────

def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")

    return Groq(api_key=api_key)


def generate_response(message: str) -> str:
    client = get_client()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
    )

    return response.choices[0].message.content or ""


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Kaushix API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/api/health")
def health():
    return {"status": "healthy"}


@app.post("/api/assistant")
def assistant(request: AssistantRequest):
    try:
        response = generate_response(request.message)

        return {
            "response": response,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate response",
        ) from exc


# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────

demo = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="Message"),
    outputs=gr.Textbox(label="Response"),
    title="Kaushix Assistant",
)

app = gr.mount_gradio_app(
    app,
    demo,
    path="/gradio",
)
