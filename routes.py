import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

import services
import sessions
from schemas import AssistantRequest


logger = logging.getLogger("uvicorn.error")


def build_chat_router(agents: dict[str, dict]) -> APIRouter:

    router = APIRouter(prefix="/api")

    def _persist_turn(request: AssistantRequest, history: list[dict], result) -> None:
        """Append the completed turn (user + assistant) to the session."""

        if not request.session_id:
            return

        response_text = result.get("response", "") if isinstance(result, dict) else result

        conversations = history + [
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": response_text},
        ]
        sessions.Sessions.update(request.session_id, conversations, request.ui_state)

    def chat_response(agent_name: str, request: AssistantRequest, stream: bool):

        # The session seam is opt-in: normal requests (no session_id) keep the
        # exact baseline behavior. Only client requests that carry a session
        # anchor get server-authoritative history + on-screen context.
        stored = (
            sessions.Sessions.get(request.session_id) if request.session_id else None
        )

        if stored is not None and stored.get("conversation"):
            history = list(stored["conversation"])
        else:
            history = [turn.model_dump() for turn in request.history]

        screen = (
            request.ui_state or (stored or {}).get("ui_state")
        ) if request.session_id else None

        if stream:
            return StreamingResponse(
                services.stream_chat_response(
                    request.message, agent_name, history, screen=screen
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )

        try:
            if request.session_id:
                result = services.generate_chat_response(
                    request.message, agent_name, history=history, screen=screen
                )

                _persist_turn(request, history, result)

                return {
                    "session_id": request.session_id,
                    "response": result.get("response", "") if isinstance(result, dict) else result,
                    "tool_calls": result.get("tool_calls", []) if isinstance(result, dict) else [],
                }

            result = services.generate_chat_response(request.message, agent_name, history)

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
