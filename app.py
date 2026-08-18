import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agents import AGENTS, MODEL_PROMPTS, MODELS
from routes import build_chat_router
from services import generate_chat_response, get_client, stream_chat_response

load_dotenv()


app = FastAPI(
    title="Kaushix API",
    version="0.4.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(build_chat_router(AGENTS))


@app.get("/")
def root():
    return {
        "name": "Kaushix API",
        "version": "0.4.0",
        "status": "running",
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
