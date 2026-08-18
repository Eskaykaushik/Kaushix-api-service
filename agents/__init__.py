import importlib
import pkgutil

__all__ = ["AGENTS", "MODELS", "MODEL_PROMPTS"]

DEFAULT_PROVIDER = "groq"


def _discover() -> dict[str, dict]:
    agents: dict[str, dict] = {}

    for module_info in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
        if module_info.name.startswith("_"):
            continue

        module = importlib.import_module(f"{__name__}.{module_info.name}")

        agents[module_info.name] = {
            "model": module.MODEL,
            "temperature": module.TEMPERATURE,
            "max_tokens": module.MAX_TOKENS,
            "prompt": module.PROMPT,
            "label": getattr(module, "LABEL", module_info.name.capitalize()),
            "fallbacks": getattr(module, "FALLBACKS", []),
            "provider": getattr(module, "PROVIDER", DEFAULT_PROVIDER),
            "tools": getattr(module, "TOOLS", None),
            "run_tool": getattr(module, "run_tool", None),
        }

    return agents


def _validate(agents: dict[str, dict]) -> None:
    for name, spec in agents.items():
        for fallback in spec["fallbacks"]:
            if fallback not in agents:
                raise ValueError(
                    f"Agent '{name}' references unknown fallback '{fallback}'"
                )

            if agents[fallback]["provider"] != spec["provider"]:
                raise ValueError(
                    f"Agent '{name}' fallback '{fallback}' uses provider "
                    f"'{agents[fallback]['provider']}' but '{name}' uses "
                    f"'{spec['provider']}'; fallbacks must share a provider"
                )


AGENTS = _discover()

_validate(AGENTS)

MODELS = {name: spec["model"] for name, spec in AGENTS.items()}

MODEL_PROMPTS = {name: spec["prompt"] for name, spec in AGENTS.items()}
