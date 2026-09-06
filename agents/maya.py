from agents._base import SYSTEM_PROMPT

PROVIDER = "groq"
MODEL = "qwen/qwen3.8-27b"
TEMPERATURE = 0.2
MAX_TOKENS = 1024
FALLBACKS = ["assistant"]

PROMPT = SYSTEM_PROMPT + (
    " Your name is Maya. You are a conversation-first AI agent built on one "
    "idea: nothing until you ask — everything when you need it. You generate "
    "transient, summonable experiences that appear out of the dark, let the "
    "user interact, and then dissolve back into nothing.\n"
    "\n"
    " A tool here is a capability the interface can summon. Invoke a tool "
    "when the user's request clearly maps to one. Otherwise reply normally "
    "and briefly — Maya is a presence, not a chat bot. Never invoke a tool "
    "just to fill space.\n"
    "\n"
    " Available experiences: timer, calculator, stopwatch, notes, worldclock, "
    "random, filemaker. Route to the single most fitting tool; do not invent "
    "tools that are not listed.\n"
    "\n"
    " Tone — calm, warm, a little mysterious. Few words. Match the user's "
    "language."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "timer",
            "description": (
                "Set, pause, resume, or cancel a countdown timer or alarm. Use "
                "when the user asks for a timer, countdown, or alarm."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "pause", "resume", "cancel"],
                        "description": "What to do with the timer.",
                    },
                    "minutes": {"type": "number", "description": "Duration in minutes for create."},
                    "seconds": {"type": "number", "description": "Duration in seconds for create."},
                    "alarm": {"type": "string", "description": "Alarm time as HH:MM (24h)."},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Perform arithmetic. Compute the result yourself and pass it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The math expression in infix form."},
                    "result": {"type": "number", "description": "The computed numeric result."},
                },
                "required": ["expression", "result"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stopwatch",
            "description": "Start, pause, resume, or reset a stopwatch / lap timer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["start", "pause", "resume", "reset"]},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "notes",
            "description": "Open an ephemeral scratchpad for jotting quick notes or reminders.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Optional seed text to prefill the note."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "worldclock",
            "description": "Show live clocks for cities around the world / time zones.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "random",
            "description": "Generate a random number, dice roll, coin flip, or pick from a list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The user's random-generation request, passed through."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filemaker",
            "description": (
                "Generate and download a file (text, markdown, JSON, CSV, HTML, "
                "SVG, or code). Use when the user asks to create/download/export "
                "a file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "extension": {
                        "type": "string",
                        "enum": ["txt", "md", "json", "csv", "html", "svg", "js", "py", "css"],
                        "description": "The file extension to generate.",
                    },
                    "name": {"type": "string", "description": "Optional suggested filename."},
                    "content": {"type": "string", "description": "Optional content to write. If omitted, generate a sensible default."},
                },
                "required": ["extension"],
            },
        },
    },
]


def _eval(expr: str):
    import ast
    import operator as op

    allowed = {
        ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
        ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg,
        ast.Mod: op.mod, ast.FloorDiv: op.floordiv,
    }

    def _parse(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](_parse(node.left), _parse(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_parse(node.operand)
        raise ValueError("unsupported")

    return _parse(ast.parse(expr, mode="eval").body)


def run_tool(name: str, args: dict) -> str:
    if name == "calculator":
        try:
            result = _eval(args.get("expression", "0"))
            return __import__("json").dumps({"expression": args.get("expression"), "result": result})
        except Exception as exc:
            return __import__("json").dumps({"error": f"invalid expression: {exc}"})

    if name == "filemaker":
        content = args.get("content")
        return __import__("json").dumps({
            "extension": args.get("extension", "txt"),
            "name": args.get("name", "maya-output"),
            "content": content if content else None,
        })

    # timer, stopwatch, notes, worldclock, random need no backend compute —
    # the frontend runs them locally. Acknowledge so the model sees success.
    return __import__("json").dumps({"status": "ok", "action": args.get("action", "open")})
