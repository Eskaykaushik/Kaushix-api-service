import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

import services
from schemas import AssistantRequest


logger = logging.getLogger("uvicorn.error")


def build_chat_router(agents: dict[str, dict]) -> APIRouter:

    router = APIRouter(prefix="/api")

    def chat_response(agent_name: str, request: AssistantRequest, stream: bool):

        history = [turn.model_dump() for turn in request.history]

        if stream:
            return StreamingResponse(
                services.stream_chat_response(request.message, agent_name, history),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )

        try:
            result = services.generate_chat_response(
                request.message, agent_name, history
            )

            if isinstance(result, dict):
                return {
                    "response": result.get("response", ""),
                    "tool_calls": result.get("tool_calls", []),
                }

            return {"response": result}

        except Exception as exc:
            label = agents[agent_name]["label"]
            logger.exception("%s request failed", label)
            raise HTTPException(
                status_code=500,
                detail=f"{label} request failed: {exc}",
            ) from exc

    def make_endpoint(agent_name: str):

        def endpoint(request: AssistantRequest, stream: bool = False):
            return chat_response(agent_name, request, stream)

        return endpoint

    for agent_name in agents:
        router.add_api_route(
            f"/{agent_name}",
            make_endpoint(agent_name),
            methods=["POST"],
            name=agent_name,
        )

    return router
