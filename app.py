import logging
import os

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel


logger = logging.getLogger("uvicorn.error")


# ==========================================
# Configuration
# ==========================================

MODELS = {
    "assistant": "openai/gpt-oss-20b",
    "reason": "openai/gpt-oss-120b",
    "fast": "openai/gpt-oss-20b",
    "research": "groq/compound-mini",
    "compound": "groq/compound",
}


# ==========================================
# App
# ==========================================

app = FastAPI(
    title="Kaushix API",
    version="0.2.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Schema
# ==========================================

class AssistantRequest(BaseModel):
    message: str


# ==========================================
# Groq
# ==========================================

def get_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")

    return Groq(api_key=api_key)


def generate_response(message: str, model: str) -> str:

    client = get_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": message,
            }
        ],
    )

    return response.choices[0].message.content or ""


# ==========================================
# Routes
# ==========================================

@app.get("/")
def root():
    return {
        "name": "Kaushix API",
        "version": "0.2.0",
        "status": "running",
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
    }


@app.post("/api/assistant")
def assistant(request: AssistantRequest):

    try:
        return {
            "response": generate_response(
                request.message,
                MODELS["assistant"],
            )
        }

    except Exception as exc:
        logger.exception("Assistant request failed")
        raise HTTPException(
            status_code=500,
            detail=f"Assistant request failed: {exc}",
        ) from exc


@app.post("/api/reason")
def reason(request: AssistantRequest):

    try:
        return {
            "response": generate_response(
                request.message,
                MODELS["reason"],
            )
        }

    except Exception as exc:
        logger.exception("Reasoning request failed")
        raise HTTPException(
            status_code=500,
            detail=f"Reasoning request failed: {exc}",
        ) from exc


@app.post("/api/fast")
def fast(request: AssistantRequest):

    try:
        return {
            "response": generate_response(
                request.message,
                MODELS["fast"],
            )
        }

    except Exception as exc:
        logger.exception("Fast request failed")
        raise HTTPException(
            status_code=500,
            detail=f"Fast request failed: {exc}",
        ) from exc


@app.post("/api/research")
def research(request: AssistantRequest):

    try:
        return {
            "response": generate_response(
                request.message,
                MODELS["research"],
            )
        }

    except Exception as exc:
        logger.exception("Research request failed")
        raise HTTPException(
            status_code=500,
            detail=f"Research request failed: {exc}",
        ) from exc


@app.post("/api/compound")
def compound(request: AssistantRequest):

    try:
        return {
            "response": generate_response(
                request.message,
                MODELS["compound"],
            )
        }

    except Exception as exc:
        logger.exception("Compound request failed")
        raise HTTPException(
            status_code=500,
            detail=f"Compound request failed: {exc}",
        ) from exc


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
