from fastapi import FastAPI
import gradio as gr

app = FastAPI(title="Kaushix API")


@app.get("/")
def root():
    return {
        "name": "Kaushix API",
        "status": "running"
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy"
    }


demo = gr.Interface(
    lambda: "Kaushix API is running",
    inputs=None,
    outputs="text",
)


app = gr.mount_gradio_app(app, demo, path="/gradio")
