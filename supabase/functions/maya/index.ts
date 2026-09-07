// Supabase Edge Function: maya
//
// Thin HTTP handler over `./_lib.ts`, the faithful Deno port of the FastAPI
// `POST /api/maya` tool loop. Behavior mirrors Render's endpoint; sessions
// are client-supplied (maya.js ships full history every turn), so this
// runtime keeps no state.

import {
  buildMessages,
  completeWithTools,
  FALLBACK_MODEL,
  MAX_TOKENS,
  MODEL,
  TEMPERATURE,
  TOOLS,
} from "./_lib.ts";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
  "Access-Control-Max-Age": "86400",
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...CORS },
  });
}

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: CORS });
  }

  if (req.method === "GET") {
    return json({
      name: "Kaushix API",
      version: "0.4.0-edge",
      status: "running",
      source: "supabase",
    });
  }

  if (req.method !== "POST") {
    return json({ detail: "Method not allowed" }, 405);
  }

  let body: {
    message?: unknown;
    history?: unknown;
    session_id?: unknown;
    ui_state?: unknown;
  };
  try {
    body = await req.json();
  } catch {
    return json({ detail: "Invalid JSON body" }, 400);
  }

  if (typeof body.message !== "string" || !body.message.trim()) {
    return json({ detail: "message is required" }, 400);
  }

  const history: { role: string; content: string }[] =
    Array.isArray(body.history)
      ? (body.history as unknown[]).filter(
        (t): t is { role: string; content: string } =>
          typeof t === "object" && t !== null &&
          typeof (t as Record<string, unknown>).role === "string" &&
          typeof (t as Record<string, unknown>).content === "string",
      )
      : [];

  try {
    const messages = buildMessages(body.message, history, body.ui_state);
    const out = (result: { response: string; toolCalls: unknown[] }) => {
      const r: Record<string, unknown> = {
        response: result.response,
        tool_calls: result.toolCalls,
      };
      if (typeof body.session_id === "string" && body.session_id) {
        r.session_id = body.session_id;
      }
      return r;
    };

    // FALLBACKS = ["assistant"]: try the primary model, then the fallback.
    try {
      return json(
        out(
          await completeWithTools(
            MODEL,
            messages,
            TEMPERATURE,
            MAX_TOKENS,
            TOOLS as object[],
          ),
        ),
      );
    } catch (exc) {
      console.error("primary model failed:", exc);
      return json(
        out(
          await completeWithTools(
            FALLBACK_MODEL,
            messages,
            TEMPERATURE,
            MAX_TOKENS,
            TOOLS as object[],
          ),
        ),
      );
    }
  } catch (exc) {
    return json({
      detail: `maya request failed: ${
        exc instanceof Error ? exc.message : String(exc)
      }`,
    }, 502);
  }
});
