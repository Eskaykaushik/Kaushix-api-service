import os

import httpx

RESEND_URL = "https://api.resend.com/emails"


def _split(value):
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _build_html(name, email, company, rows_html, total, notes):
    customer_line = f"{name} &lt;{email}&gt;"
    if company:
        customer_line += f" — {company}"

    header = (
        '<tr><td style="background:#1a1a2e;padding:24px 32px;">'
        '<table width="100%" cellpadding="0" cellspacing="0"><tr>'
        '<td style="color:#ffffff;font-size:20px;font-weight:700;letter-spacing:0.5px;">ElectroSemi</td>'
        '<td align="right" style="color:#a0a0b8;font-size:13px;">New Order</td>'
        "</tr></table>"
        "</td></tr>"
    )

    customer = (
        '<p style="margin:0 0 4px;font-size:12px;text-transform:uppercase;letter-spacing:1px;color:#8888a0;">Customer</p>'
        f'<p style="margin:0 0 24px;font-size:15px;color:#1a1a2e;">{customer_line}</p>'
    )

    table = (
        '<table width="100%" cellpadding="0" cellspacing="0" style="border:1px solid #e8e8ef;border-radius:6px;overflow:hidden;">'
        "<thead><tr>"
        '<th style="background:#f0f0f7;padding:10px 14px;font-size:12px;text-transform:uppercase;letter-spacing:0.5px;color:#6b6b80;text-align:left;">SKU</th>'
        '<th style="background:#f0f0f7;padding:10px 14px;font-size:12px;text-transform:uppercase;letter-spacing:0.5px;color:#6b6b80;text-align:left;">Item</th>'
        '<th style="background:#f0f0f7;padding:10px 14px;font-size:12px;text-transform:uppercase;letter-spacing:0.5px;color:#6b6b80;text-align:right;">Qty</th>'
        '<th style="background:#f0f0f7;padding:10px 14px;font-size:12px;text-transform:uppercase;letter-spacing:0.5px;color:#6b6b80;text-align:right;">Unit Price</th>'
        '<th style="background:#f0f0f7;padding:10px 14px;font-size:12px;text-transform:uppercase;letter-spacing:0.5px;color:#6b6b80;text-align:right;">Total</th>'
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table>"
    )

    total_block = (
        '<table width="100%" cellpadding="0" cellspacing="0" style="margin-top:16px;">'
        '<tr><td align="right" style="padding:12px 14px;background:#f9f9fc;border-radius:6px;">'
        '<span style="font-size:13px;color:#6b6b80;text-transform:uppercase;letter-spacing:0.5px;">Grand Total</span>&nbsp;&nbsp;'
        f'<span style="font-size:20px;font-weight:700;color:#1a1a2e;">${total:,.2f}</span>'
        "</td></tr></table>"
    )

    notes_block = ""
    if notes:
        notes_block = (
            '<p style="margin:24px 0 0;font-size:13px;color:#6b6b80;">'
            f'<strong style="color:#1a1a2e;">Notes:</strong> {notes}</p>'
        )

    footer = (
        '<tr><td style="padding:20px 32px;background:#f9f9fc;border-top:1px solid #e8e8ef;">'
        '<p style="margin:0;font-size:11px;color:#a0a0b8;text-align:center;">'
        "ElectroSemi — Powered by Kaushix Labs</p>"
        "</td></tr>"
    )

    return (
        "<!DOCTYPE html>"
        '<html><head><meta charset="utf-8"></head>'
        '<body style="margin:0;padding:0;background:#f4f5f7;font-family:Arial,Helvetica,sans-serif;">'
        '<table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f5f7;padding:32px 0;">'
        '<tr><td align="center">'
        '<table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.08);">'
        + header
        + '<tr><td style="padding:32px;">'
        + customer
        + table
        + total_block
        + notes_block
        + "</td></tr>"
        + footer
        + "</table>"
        "</td></tr></table>"
        "</body></html>"
    )


def _build_text(name, email, company, items, total, notes):
    sep = "-" * 44
    header = (
        f"NEW ELECTROSEMI ORDER\n"
        f"{sep}\n"
        f"Customer : {name} <{email}>"
    )
    if company:
        header += f"\nCompany  : {company}"

    lines = [header, sep, ""]
    for item in items:
        qty = int(item.get("quantity", 0) or 0)
        price = float(item.get("unitPrice", 0) or 0)
        lines.append(
            f"  {item.get('sku', ''):<14}{item.get('name', ''):<20}"
            f"{qty:>5}  x  ${price:>9.2f}  =  ${qty * price:>10.2f}"
        )

    lines += [
        "",
        sep,
        f"  {'GRAND TOTAL':>46}  ${total:>10.2f}",
        sep,
    ]

    if notes:
        lines += ["", f"Notes: {notes}"]

    lines += ["", "— ElectroSemi / Kaushix Labs —"]
    return "\n".join(lines)


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
            f'<td style="padding:10px 14px;border-bottom:1px solid #f0f0f7;font-size:14px;color:#1a1a2e;">{item.get("sku", "")}</td>'
            f'<td style="padding:10px 14px;border-bottom:1px solid #f0f0f7;font-size:14px;color:#1a1a2e;">{item.get("name", "")}</td>'
            f'<td style="padding:10px 14px;border-bottom:1px solid #f0f0f7;font-size:14px;color:#1a1a2e;text-align:right;">{qty}</td>'
            f'<td style="padding:10px 14px;border-bottom:1px solid #f0f0f7;font-size:14px;color:#1a1a2e;text-align:right;">${price:.2f}</td>'
            f'<td style="padding:10px 14px;border-bottom:1px solid #f0f0f7;font-size:14px;color:#1a1a2e;text-align:right;font-weight:600;">${line_total:.2f}</td>'
            "</tr>"
        )
    rows_html = "".join(rows)

    html = _build_html(name, email, company, rows_html, total, notes)
    text = _build_text(name, email, company, items, total, notes)

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
