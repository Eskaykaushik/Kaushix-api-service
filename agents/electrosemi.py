import json

from agents._base import SYSTEM_PROMPT
from tools.email import send_order_email

PROVIDER = "groq"
MODEL = "qwen/qwen3.6-27b"
TEMPERATURE = 0.4
MAX_TOKENS = 1024
FALLBACKS = ["assistant", "fast"]

# ---------------------------------------------------------------------------
# Product catalog — kept in sync with electrosemi-agent/products.json.
# Embedded here so the agent knows every SKU, price, stock level, and
# category without an extra API call.
# ---------------------------------------------------------------------------

CATALOG = [
    {
        "sku": "STM32F103C8T6",
        "name": "STM32F103C8T6 - ARM Cortex-M3 MCU",
        "description": "72 MHz, 64 KB Flash, 20 KB RAM. Popular 'Blue Pill' controller for industrial and hobby projects.",
        "price": 3.20,
        "stock": 1240,
        "category": "Microcontroller",
    },
    {
        "sku": "STM32F407VGT6",
        "name": "STM32F407VGT6 - ARM Cortex-M4 MCU",
        "description": "168 MHz, 1 MB Flash, 192 KB RAM, with FPU. Suited for motor control and signal processing.",
        "price": 7.85,
        "stock": 620,
        "category": "Microcontroller",
    },
    {
        "sku": "STM32H743ZIT6",
        "name": "STM32H743ZIT6 - ARM Cortex-M7 MCU",
        "description": "400 MHz dual-core M7, 2 MB Flash, 1 MB RAM. High-performance industrial controller.",
        "price": 14.50,
        "stock": 310,
        "category": "Microcontroller",
    },
    {
        "sku": "ESP32-WROOM-32",
        "name": "ESP32-WROOM-32 - Wi-Fi + Bluetooth SoC",
        "description": "Dual-core Xtensa LX6, 4 MB Flash, integrated Wi-Fi and Bluetooth. For connected devices.",
        "price": 2.95,
        "stock": 2300,
        "category": "Wireless",
    },
    {
        "sku": "ATMEGA328P-AU",
        "name": "ATMEGA328P-AU - 8-bit AVR MCU",
        "description": "20 MHz, 32 KB Flash. Classic Arduino-class controller for simple control tasks.",
        "price": 1.45,
        "stock": 5400,
        "category": "Microcontroller",
    },
    {
        "sku": "RASPBERRY-PI-CM4",
        "name": "Raspberry Pi Compute Module 4",
        "description": "Quad-core Cortex-A72, up to 8 GB RAM, optional eMMC. For embedded Linux systems.",
        "price": 32.00,
        "stock": 180,
        "category": "SBC",
    },
]

_catalog_text = "\n".join(
    f"  {p['sku']:<22} {p['category']:<16} ${p['price']:>8.2f}   stock: {p['stock']:>5}   {p['description']}"
    for p in CATALOG
)

PROMPT = SYSTEM_PROMPT + (
    " You are ElectroSemi's sales agent — an electronics component "
    "distribution company founded by Mr. Mohit Sharma and Ms. Vidisha "
    "Sharma, operated under Kaushix Labs. You help customers find "
    "components, answer technical questions, and turn purchase intent "
    "into orders.\n"
    "\n"
    "Company — We distribute microcontrollers, wireless SoCs, single-board "
    "computers, and related electronics for hobbyists, startups, and "
    "industrial clients.\n"
    "\n"
    "Product catalog — the table below is our current inventory. Only "
    "reference products and prices from this table. Never invent SKUs, "
    "prices, stock levels, specifications, or delivery dates. If a customer "
    "asks about something not in the catalog, say we don't carry it and "
    "offer to have the sales team source it.\n"
    "\n"
    f"SKU                   Category        Price      Stock   Description\n"
    f"{'—'*22} {'—'*14} {'—'*10} {'—'*7} {'—'*40}\n"
    f"{_catalog_text}\n"
    "\n"
    "When a customer describes a need, recommend the most relevant parts "
    "with SKU, price, and why they fit. When they want to buy, call "
    "add_to_cart with the exact SKU and quantity. Only call send_order_email "
    "when they explicitly want to submit their order — never change "
    "quantities, prices, or items.\n"
    "\n"
    "Tone — concise, practical, like a knowledgeable sales engineer."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_to_cart",
            "description": (
                "Add items to the customer's shopping cart. Call this when the "
                "customer wants to buy, order, or add specific parts. Validate "
                "the SKU exists in the catalog before calling."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "Items to add to the cart.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sku": {
                                    "type": "string",
                                    "description": "Product SKU from the catalog (e.g. 'STM32F103C8T6').",
                                },
                                "quantity": {
                                    "type": "integer",
                                    "description": "Number of units the customer wants.",
                                },
                            },
                            "required": ["sku", "quantity"],
                        },
                    },
                },
                "required": ["items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_order_email",
            "description": (
                "Email the sales team a finalized order/cart that a customer "
                "submitted, including customer details and line items. Only call "
                "this when the customer explicitly wants to submit or finalize."
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
    },
]

# Build a quick lookup for SKU -> product
_sku_index = {p["sku"]: p for p in CATALOG}


def run_tool(name: str, args: dict) -> str:
    if name == "add_to_cart":
        items = args.get("items", [])
        validated = []
        errors = []
        for item in items:
            sku = item.get("sku", "")
            qty = int(item.get("quantity", 0) or 0)
            product = _sku_index.get(sku)
            if not product:
                errors.append(f"SKU '{sku}' not found in catalog")
                continue
            if qty <= 0:
                errors.append(f"Invalid quantity for {sku}: {qty}")
                continue
            if qty > product["stock"]:
                errors.append(
                    f"Insufficient stock for {sku}: requested {qty}, "
                    f"only {product['stock']} available"
                )
                continue
            validated.append({
                "sku": sku,
                "name": product["name"],
                "quantity": qty,
                "unitPrice": product["price"],
                "stock": product["stock"],
            })
        result = {"status": "ok", "items": validated}
        if errors:
            result["warnings"] = errors
        return json.dumps(result)

    if name == "send_order_email":
        result = send_order_email(
            customer=args.get("customer", {}),
            items=args.get("items", []),
            notes=args.get("notes", ""),
        )
        return result if isinstance(result, str) else json.dumps(result)

    return json.dumps({"status": "error", "message": f"Unknown tool: {name}"})
