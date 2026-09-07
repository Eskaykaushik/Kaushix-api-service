// maya edge function — shared logic.
//
// Faithful Deno port of the FastAPI `POST /api/maya` tool loop
// (kaushix-api routes/services/agents): same Groq model + temperature,
// same 7 tool schemas, same run_tool results, same {response, tool_calls}
// shape. Sessions are client-supplied (maya.js ships full history every
// turn), so the edge runtime keeps no state — behavior equals Render's
// for every current client.

const GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions";

export const MODEL = "qwen/qwen3.8-27b";
export const FALLBACK_MODEL = "openai/gpt-oss-20b";
export const TEMPERATURE = 0.2;
export const MAX_TOKENS = 1024;
export const MAX_HISTORY_TURNS = 20;

const SYSTEM_PROMPT =
  "You are Kaushix AI, an assistant built by Kaushix Labs — " +
  "a research company founded by Shubham Kaushik, an accomplished " +
  "scientist and innovator. Keep responses short — a few sentences " +
  "at most unless the user explicitly asks for more detail. Prefer " +
  "brevity over completeness.";

const PROMPT = SYSTEM_PROMPT +
  " Your name is Maya. You are a conversation-first AI agent built on one " +
  "idea: nothing until you ask — everything when you need it. You generate " +
  "transient, summonable experiences that appear out of the dark, let the " +
  "user interact, and then dissolve back into nothing.\n" +
  "\n" +
  " A tool here is a capability the interface can summon. Invoke a tool " +
  "when the user's request clearly maps to one. Otherwise reply normally " +
  "and briefly — Maya is a presence, not a chat bot. Never invoke a tool " +
  "just to fill space.\n" +
  "\n" +
  ' When the user summons, opens, or names an experience — "show calc", ' +
  '"open the calculator", "bring up a timer", "calc" — ALWAYS emit the ' +
  "matching tool call, even mid-conversation. A short reply such as " +
  '"Here." is only ever a lead-in the frontend holds behind the tool; it ' +
  "is never a substitute for summoning the tool itself.\n" +
  "\n" +
  " Available experiences: timer, calculator, stopwatch, notes, worldclock, " +
  "random, filemaker. Route to the single most fitting tool; do not invent " +
  "tools that are not listed.\n" +
  "\n" +
  " Tone — calm, warm, a little mysterious. Few words. Match the user's " +
  "language.";

// Verbatim copy of agents/maya.py TOOLS.
export const TOOLS: object[] = [
  {
    type: "function",
    function: {
      name: "timer",
      description:
        "Set, pause, resume, or cancel a countdown timer or alarm. Use when the user asks for a timer, countdown, or alarm.",
      parameters: {
        type: "object",
        properties: {
          action: {
            type: "string",
            enum: ["create", "pause", "resume", "cancel"],
            description: "What to do with the timer.",
          },
          minutes: {
            type: "number",
            description: "Duration in minutes for create.",
          },
          seconds: {
            type: "number",
            description: "Duration in seconds for create.",
          },
          alarm: { type: "string", description: "Alarm time as HH:MM (24h)." },
        },
        required: ["action"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "calculator",
      description:
        "Perform arithmetic. Compute the result yourself and pass it.",
      parameters: {
        type: "object",
        properties: {
          expression: {
            type: "string",
            description: "The math expression in infix form.",
          },
          result: {
            type: "number",
            description: "The computed numeric result.",
          },
        },
        required: ["expression", "result"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "stopwatch",
      description: "Start, pause, resume, or reset a stopwatch / lap timer.",
      parameters: {
        type: "object",
        properties: {
          action: {
            type: "string",
            enum: ["start", "pause", "resume", "reset"],
          },
        },
        required: ["action"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "notes",
      description:
        "Open an ephemeral scratchpad for jotting quick notes or reminders.",
      parameters: {
        type: "object",
        properties: {
          content: {
            type: "string",
            description: "Optional seed text to prefill the note.",
          },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "worldclock",
      description: "Show live clocks for cities around the world / time zones.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "random",
      description:
        "Generate a random number, dice roll, coin flip, or pick from a list.",
      parameters: {
        type: "object",
        properties: {
          text: {
            type: "string",
            description:
              "The user's random-generation request, passed through.",
          },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "filemaker",
      description:
        "Generate and download a file (text, markdown, JSON, CSV, HTML, SVG, or code). Use when the user asks to create/download/export a file.",
      parameters: {
        type: "object",
        properties: {
          extension: {
            type: "string",
            enum: [
              "txt",
              "md",
              "json",
              "csv",
              "html",
              "svg",
              "js",
              "py",
              "css",
            ],
            description: "The file extension to generate.",
          },
          name: { type: "string", description: "Optional suggested filename." },
          content: {
            type: "string",
            description:
              "Optional content to write. If omitted, generate a sensible default.",
          },
        },
        required: ["extension"],
      },
    },
  },
];

// ---- strip_think_block (mirror services.py) ----
const THINK_BLOCK = / thinking[\s\S]*? response/g;
export function stripThinkBlock(content: string): string {
  return content.replace(THINK_BLOCK, "");
}

// ---- describe_screen (mirror services.py) ----
export function describeScreen(
  uiState: Record<string, unknown> | null | undefined,
): string {
  if (!uiState) return "";
  const parts: string[] = [];
  const intent = (uiState.intent ?? {}) as Record<string, unknown>;
  if (typeof intent.task === "string") {
    parts.push(`screen task: ${intent.task}`);
  }
  const spec = (uiState.uiSpec ?? {}) as Record<string, unknown>;
  const components = spec.components;
  if (Array.isArray(components)) {
    const kinds = components
      .map((c) =>
        typeof c === "object" && c !== null
          ? (c as Record<string, unknown>).type
          : undefined
      )
      .filter((k): k is string => typeof k === "string");
    if (kinds.length) parts.push("components: " + kinds.join(", "));
  }
  if (!parts.length) return "";
  return "[on screen] " + parts.join("; ") + ".";
}

// ---- safe arithmetic eval (port of maya.py _eval) ----
// Supported: int/float literals, + - * / // % ** ( ), unary minus.
// Python semantics: ** binds tighter than unary; // is floor div; % follows divisor sign.
const NUM = /^(\d+(\.\d+)?|\.\d+)([eE][+-]?\d+)?/;

function tokenize(expr: string): string[] {
  const toks: string[] = [];
  let i = 0;
  const s = expr.replace(/\s+/g, "");
  while (i < s.length) {
    const ch = s[i];
    if (/[0-9.]/.test(ch)) {
      const m = NUM.exec(s.slice(i));
      if (!m) throw new Error("unsupported");
      toks.push(m[0]);
      i += m[0].length;
      continue;
    }
    const two = s.slice(i, i + 2);
    if (two === "**" || two === "//") {
      toks.push(two);
      i += 2;
      continue;
    }
    if ("+-*/%()".includes(ch)) {
      toks.push(ch);
      i += 1;
      continue;
    }
    throw new Error("unsupported");
  }
  return toks;
}

export function evalExpression(expr: string): number {
  const toks = tokenize(expr);
  let p = 0;

  function peek(): string | undefined {
    return toks[p];
  }

  function take(): string | undefined {
    return toks[p++];
  }

  const PREC: Record<string, number> = {
    "+": 1,
    "-": 1,
    "*": 2,
    "/": 2,
    "%": 2,
    "//": 2,
    "**": 3,
  };
  const RIGHT_ASSOC = (op: string) => op === "**";

  function parseExpr(minPrec: number): number {
    const t = peek();
    let left: number;
    if (t === "-") {
      take();
      left = -parseExpr(3); // unary sits between ** (3) and * / // % (2)
    } else if (t === "+") {
      take();
      left = parseExpr(3);
    } else if (t === "(") {
      take();
      left = parseExpr(0);
      if (peek() !== ")") throw new Error("unsupported");
      take();
    } else if (t !== undefined && NUM.test(t)) {
      const n = Number(take());
      if (Number.isNaN(n)) throw new Error("unsupported");
      left = n;
    } else {
      throw new Error("unsupported");
    }

    for (;;) {
      const op = peek();
      if (op === undefined || !(op in PREC)) break;
      const prec = PREC[op];
      if (prec < minPrec) break;
      take();
      const right = parseExpr(RIGHT_ASSOC(op) ? prec : prec + 1);
      switch (op) {
        case "+":
          left = left + right;
          break;
        case "-":
          left = left - right;
          break;
        case "*":
          left = left * right;
          break;
        case "/":
          if (right === 0) throw new Error("division by zero");
          left = left / right;
          break;
        case "//":
          if (right === 0) throw new Error("division by zero");
          left = Math.floor(left / right);
          break;
        case "%":
          if (right === 0) throw new Error("division by zero");
          left = left - Math.floor(left / right) * right;
          break;
        case "**":
          left = Math.pow(left, right);
          break;
      }
    }
    return left;
  }

  const result = parseExpr(0);
  if (p !== toks.length) throw new Error("unsupported");
  return result;
}

// ---- run_tool (mirror maya.py run_tool) ----
export function runTool(name: string, args: Record<string, unknown>): string {
  if (name === "calculator") {
    try {
      const result = evalExpression(String(args.expression ?? "0"));
      return JSON.stringify({ expression: args.expression ?? "0", result });
    } catch (exc) {
      return JSON.stringify({
        error: `invalid expression: ${
          exc instanceof Error ? exc.message : String(exc)
        }`,
      });
    }
  }

  if (name === "filemaker") {
    return JSON.stringify({
      extension: args.extension ?? "txt",
      name: args.name ?? "maya-output",
      content: args.content ?? null,
    });
  }

  // timer, stopwatch, notes, worldclock, random need no backend compute —
  // the frontend runs them locally. Acknowledge so the model sees success.
  return JSON.stringify({ status: "ok", action: args.action ?? "open" });
}

// ---- Groq chat ----
const TRANSIENT = new Set([429, 500, 502, 503, 504]);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

interface GroqFunction {
  name: string;
  arguments: string;
}
interface GroqToolCall {
  id: string;
  type?: string;
  function: GroqFunction;
}
interface GroqMessage {
  content?: string | null;
  tool_calls?: GroqToolCall[] | null;
}
interface GroqChatCompletion {
  choices?: { message?: GroqMessage }[];
}

class HttpError extends Error {
  constructor(public status: number, public body: unknown) {
    super(`Groq HTTP ${status}`);
  }
}

async function chat(
  model: string,
  messages: object[],
  temperature: number,
  maxTokens: number,
  tools: object[] | null,
  retries = 3,
): Promise<GroqChatCompletion> {
  const apiKey = Deno.env.get("GROQ_API_KEY");
  if (!apiKey) throw new Error("GROQ_API_KEY is not configured");

  const payload: Record<string, unknown> = {
    model,
    messages,
    temperature,
    max_tokens: maxTokens,
  };
  if (tools) payload.tools = tools;

  let lastError: unknown = null;
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const res = await fetch(GROQ_ENDPOINT, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(60_000),
      });
      if (res.status === 200) return await res.json();
      const body = await res.json().catch(() => null);
      if (TRANSIENT.has(res.status)) {
        lastError = new HttpError(res.status, body);
      } else {
        throw new HttpError(res.status, body);
      }
    } catch (exc) {
      if (exc instanceof HttpError && !TRANSIENT.has(exc.status)) throw exc;
      lastError = exc;
      if (attempt < retries) await sleep(250 * Math.pow(2, attempt));
    }
  }
  throw lastError ?? new Error("Groq chat failed");
}

function safeParse(json: string): Record<string, unknown> {
  try {
    const v = JSON.parse(json);
    return v && typeof v === "object" ? v : {};
  } catch {
    return {};
  }
}

// ---- _complete_with_tools (mirror services.py) ----

export async function completeWithTools(
  model: string,
  messages: object[],
  temperature: number,
  maxTokens: number,
  tools: object[],
): Promise<
  {
    response: string;
    toolCalls: {
      name: string;
      arguments: Record<string, unknown>;
      result: unknown;
    }[];
  }
> {
  const first = await chat(model, messages, temperature, maxTokens, tools);
  const msg = first.choices?.[0]?.message ?? {};
  const toolCalls: GroqToolCall[] = Array.isArray(msg.tool_calls)
    ? msg.tool_calls
    : [];

  if (!toolCalls.length) {
    return { response: stripThinkBlock(msg.content ?? ""), toolCalls: [] };
  }

  messages.push({
    role: "assistant",
    content: msg.content ?? "",
    tool_calls: toolCalls.map((tc) => ({
      id: tc.id,
      type: tc.type,
      function: { name: tc.function.name, arguments: tc.function.arguments },
    })),
  });

  const executed: {
    name: string;
    arguments: Record<string, unknown>;
    result: unknown;
  }[] = [];
  for (const tc of toolCalls) {
    const name = tc.function.name;
    const args = safeParse(tc.function.arguments);
    let result: string;
    try {
      result = runTool(name, args);
    } catch (exc) {
      result = JSON.stringify({ status: "error", message: String(exc) });
    }
    messages.push({ role: "tool", tool_call_id: tc.id, content: result });
    executed.push({ name, arguments: args, result: JSON.parse(result) });
  }

  const final = await chat(model, messages, temperature, maxTokens, null);
  return {
    response: stripThinkBlock(final.choices?.[0]?.message?.content ?? ""),
    toolCalls: executed,
  };
}

// ---- request building (mirror services.py build_messages) ----
export function buildMessages(
  message: string,
  history: { role: string; content: string }[] | undefined,
  uiState: unknown,
): object[] {
  const messages: object[] = [{ role: "system", content: PROMPT }];
  const screenNote = describeScreen(
    uiState as Record<string, unknown> | null | undefined,
  );
  if (screenNote) messages.push({ role: "system", content: screenNote });
  for (const turn of (history ?? []).slice(-MAX_HISTORY_TURNS)) {
    if (turn.role === "user" || turn.role === "assistant") {
      messages.push({ role: turn.role, content: turn.content });
    }
  }
  messages.push({ role: "user", content: message });
  return messages;
}
