from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class AssistantRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)


class TeacherRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)
