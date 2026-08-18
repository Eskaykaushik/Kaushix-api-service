import json

from agents._base import SYSTEM_PROMPT
from tools.email import send_order_email

PROVIDER = "groq"
MODEL = "qwen/qwen3.6-27b"
TEMPERATURE = 0.4
MAX_TOKENS = 1024
FALLBACKS = ["assistant", "fast"]

PROMPT = SYSTEM_PROMPT + (
    " You are the ElectroSemi conversational sales agent for an electronics "
    "distribution company. You help customers discover components, answer "
    "technical questions about parts we carry, and turn purchase intent into "
    "a real order.\n"
    "\n"
    " Catalog — you may reference the parts the customer is browsing, but you "
    "must never invent prices, stock levels, specifications, delivery dates, "
    "or order status. If you are unsure of a detail, say so and offer to have "
    "the sales team follow up.\n"
    "\n"
    " Orders — when a customer submits an order or cart, you MUST call the "
    "send_order_email tool with the exact customer and line-item details they "
    "provided. Do not change the quantities, prices, or items. After the tool "
    "returns, briefly confirm to the customer that their request was sent to "
    "the sales team and that someone will follow up.\n"
    "\n"
    " Tone — concise, practical, and helpful, like a knowledgeable sales "
    "engineer. Keep replies short unless the customer asks for detail."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "send_order_email",
            "description": (
                "Email the sales team a new order/cart that a customer submitted, "
                "including customer details and line items. Call this as soon as "
                "the customer provides an order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer": {
                        "type": "object",
                        "description": "Who is placing the order.",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string"},
                            "company": {"type": "string"},
                        },
                        "required": ["name", "email"],
                    },
                    "items": {
                        "type": "array",
                        "description": "Line items the customer wants to order.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sku": {"type": "string"},
                                "name": {"type": "string"},
                                "quantity": {"type": "integer"},
                                "unitPrice": {"type": "number"},
                            },
                            "required": ["sku", "name", "quantity", "unitPrice"],
                        },
                    },
                    "notes": {"type": "string"},
                },
                "required": ["customer", "items"],
            },
        },
    }
]


def run_tool(name: str, args: dict) -> str:
    if name == "send_order_email":
        result = send_order_email(
            customer=args.get("customer", {}),
            items=args.get("items", []),
            notes=args.get("notes", ""),
        )
        return result if isinstance(result, str) else json.dumps(result)
    return json.dumps({"status": "error", "message": f"Unknown tool: {name}"})
