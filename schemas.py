from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class AssistantRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)
    # Session seam (Phase-1 follow-ups): the client ships a session anchor and
    # what is currently on screen; the server keeps the authoritative history.
    session_id: str | None = None
    ui_state: dict | None = None


class TeacherRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)
