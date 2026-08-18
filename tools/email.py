import os

import httpx

RESEND_URL = "https://api.resend.com/emails"


def _split(value):
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def send_order_email(customer, items, notes=""):
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        return {"status": "error", "message": "RESEND_API_KEY is not configured"}

    sender = os.getenv("RESEND_FROM", "onboarding@resend.dev")
    recipients = _split(os.getenv("SALES_TEAM_EMAIL"))
    if not recipients:
        return {"status": "error", "message": "SALES_TEAM_EMAIL is not configured"}

    customer = customer or {}
    name = customer.get("name", "Unknown")
    email = customer.get("email", "")
    company = customer.get("company", "")

    rows = []
    total = 0.0
    for item in items:
        qty = int(item.get("quantity", 0) or 0)
        price = float(item.get("unitPrice", 0) or 0)
        line_total = qty * price
        total += line_total
        rows.append(
            "<tr>"
            f"<td>{item.get('sku', '')}</td>"
            f"<td>{item.get('name', '')}</td>"
            f"<td>{qty}</td>"
            f"<td>${price:.2f}</td>"
            f"<td>${line_total:.2f}</td>"
            "</tr>"
        )
    rows_html = "".join(rows)

    customer_line = f"{name} &lt;{email}&gt;" + (f" ({company})" if company else "")

    html = (
        "<h2>New ElectroSemi order</h2>"
        f"<p><strong>Customer:</strong> {customer_line}</p>"
        "<table border=\"1\" cellpadding=\"6\" cellspacing=\"0\">"
        "<thead><tr><th>SKU</th><th>Name</th><th>Qty</th>"
        "<th>Unit</th><th>Total</th></tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
        f"<p><strong>Total:</strong> ${total:.2f}</p>"
        f"<p><strong>Notes:</strong> {notes or '—'}</p>"
    )

    text = (
        "New ElectroSemi order\n"
        f"Customer: {name} <{email}>"
        + (f" ({company})" if company else "")
        + f"\nNotes: {notes or '-'}\n\n"
        + "\n".join(
            f"- {i.get('sku', '')} {i.get('name', '')} x{int(i.get('quantity', 0) or 0)} "
            f"@ ${float(i.get('unitPrice', 0) or 0):.2f}"
            for i in items
        )
        + f"\n\nTotal: ${total:.2f}"
    )

    payload = {
        "from": sender,
        "to": recipients,
        "subject": f"New ElectroSemi order from {name}",
        "html": html,
        "text": text,
    }

    try:
        resp = httpx.post(
            RESEND_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        if resp.status_code >= 400:
            return {
                "status": "error",
                "message": f"Resend {resp.status_code}: {resp.text}",
            }
        return {"status": "sent", "id": resp.json().get("id")}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": str(exc)}
