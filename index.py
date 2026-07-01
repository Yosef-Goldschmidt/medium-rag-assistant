"""Single Vercel Python entrypoint. Routes by URL path:
    GET  /            -> minimal chat GUI (HTML)
    GET  /api/stats   -> RAG hyperparameters
    POST /api/prompt  -> answer a question with the RAG pipeline

Imports of the heavy RAG chain are done lazily inside the handlers so that
/api/stats and the GUI page work even if the model/DB clients aren't reachable.
"""
import os
import sys
import json
from http.server import BaseHTTPRequestHandler

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Medium Article RAG Assistant</title>
<style>
  :root { --fg:#1a1a1a; --muted:#666; --line:#e5e5e5; --accent:#0b5; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; color: var(--fg);
         max-width: 780px; margin: 0 auto; padding: 32px 20px 80px; line-height: 1.5; }
  h1 { font-size: 1.5rem; margin: 0 0 4px; }
  p.sub { color: var(--muted); margin: 0 0 24px; }
  .examples { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
  .chip { font-size: .82rem; border: 1px solid var(--line); background:#fafafa; color:#333;
          padding: 6px 10px; border-radius: 999px; cursor: pointer; }
  .chip:hover { border-color: var(--accent); }
  textarea { width: 100%; min-height: 70px; padding: 12px; border: 1px solid var(--line);
             border-radius: 8px; font: inherit; resize: vertical; }
  .row { display: flex; gap: 10px; align-items: center; margin-top: 10px; }
  button { background: var(--accent); color: #fff; border: 0;
           padding: 10px 18px; border-radius: 8px; font: inherit; cursor: pointer; }
  button.secondary { background: #fff; color: #333; border: 1px solid var(--line); }
  button:disabled { opacity: .5; cursor: default; }
  #answer { white-space: pre-wrap; margin-top: 24px; padding: 16px; border: 1px solid var(--line);
            border-radius: 8px; background: #fafafa; min-height: 20px; }
  #stats { margin-top: 12px; color: var(--muted); font-size: .9rem; }
  details { margin-top: 16px; }
  summary { cursor: pointer; color: var(--muted); }
  .chunk { border-top: 1px solid var(--line); padding: 10px 0; font-size: .88rem; }
  .chunk .meta { color: var(--muted); font-size: .8rem; margin-bottom: 4px; }
  pre { white-space: pre-wrap; background:#fafafa; border:1px solid var(--line);
        border-radius:6px; padding:12px; font-size:.82rem; overflow-x:auto; }
  .spin { color: var(--muted); }
</style>
</head>
<body>
  <h1>Medium Article RAG Assistant</h1>
  <p class="sub">Answers come only from a corpus of ~7,600 Medium articles.</p>

  <div class="examples" id="examples"></div>
  <textarea id="q" placeholder="Ask a question about the Medium articles..."></textarea>
  <div class="row">
    <button id="ask">Ask</button>
    <button id="statsBtn" class="secondary">Show config (/api/stats)</button>
  </div>
  <div id="stats"></div>

  <div id="answer"></div>

  <details id="ctxWrap" style="display:none">
    <summary>Retrieved context</summary>
    <div id="context"></div>
  </details>

  <details id="promptWrap" style="display:none">
    <summary>Augmented prompt (sent to the model)</summary>
    <h4>System</h4><pre id="sysPrompt"></pre>
    <h4>User</h4><pre id="userPrompt"></pre>
  </details>

<script>
const EXAMPLES = [
  "Find an article about smell training and the brain. Give the title and author.",
  "List exactly 3 articles about education. Return only the titles.",
  "I want beginner-friendly advice on building habits that stick. Which article would you recommend, and why?"
];
const examplesEl = document.getElementById("examples");
EXAMPLES.forEach(text => {
  const c = document.createElement("span");
  c.className = "chip"; c.textContent = text;
  c.onclick = () => { document.getElementById("q").value = text; };
  examplesEl.appendChild(c);
});

const askBtn = document.getElementById("ask");
const statsBtn = document.getElementById("statsBtn");
const answerEl = document.getElementById("answer");
const statsEl = document.getElementById("stats");
const ctxWrap = document.getElementById("ctxWrap");
const contextEl = document.getElementById("context");
const promptWrap = document.getElementById("promptWrap");

async function ask() {
  const question = document.getElementById("q").value.trim();
  if (!question) return;
  askBtn.disabled = true;
  answerEl.innerHTML = '<span class="spin">Thinking… (this can take 10–30s)</span>';
  ctxWrap.style.display = "none";
  promptWrap.style.display = "none";
  contextEl.innerHTML = "";
  try {
    const res = await fetch("/api/prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });
    const data = await res.json();
    if (data.error) { answerEl.textContent = "Error: " + data.error; return; }
    answerEl.textContent = data.response || "(no response)";

    if (Array.isArray(data.context) && data.context.length) {
      data.context.forEach(c => {
        const d = document.createElement("div");
        d.className = "chunk";
        d.innerHTML = '<div class="meta">' + (c.title || "") +
          "  ·  score " + (c.score != null ? c.score.toFixed(3) : "?") +
          "  ·  id " + (c.article_id || "") + "</div>" +
          (c.chunk ? c.chunk.slice(0, 320) + "…" : "");
        contextEl.appendChild(d);
      });
      ctxWrap.style.display = "block";
    }

    if (data.Augmented_prompt) {
      document.getElementById("sysPrompt").textContent = data.Augmented_prompt.System || "";
      document.getElementById("userPrompt").textContent = data.Augmented_prompt.User || "";
      promptWrap.style.display = "block";
    }
  } catch (e) {
    answerEl.textContent = "Request failed: " + e;
  } finally {
    askBtn.disabled = false;
  }
}

async function showStats() {
  statsBtn.disabled = true;
  statsEl.textContent = "Loading…";
  try {
    const res = await fetch("/api/stats");
    const data = await res.json();
    statsEl.textContent = "Config → chunk_size: " + data.chunk_size +
      "  ·  overlap_ratio: " + data.overlap_ratio + "  ·  top_k: " + data.top_k;
  } catch (e) {
    statsEl.textContent = "Failed to load config: " + e;
  } finally {
    statsBtn.disabled = false;
  }
}

askBtn.onclick = ask;
statsBtn.onclick = showStats;
</script>
</body>
</html>"""


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = self._route()
        if route == "/api/stats":
            import config
            self._send(200, {
                "chunk_size": config.CHUNK_SIZE,
                "overlap_ratio": config.OVERLAP_RATIO,
                "top_k": config.TOP_K,
            })
        elif route == "/":
            self._send_html(200, HTML_PAGE)
        else:
            self._send(404, {"error": "Not found"})

    def do_POST(self):
        if self._route() == "/api/prompt":
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b"{}"
                body = json.loads(raw or b"{}")
                question = body.get("question", "")
                if not question:
                    self._send(400, {"error": "Missing 'question' in request body."})
                    return
                from rag import answer_question
                self._send(200, answer_question(question))
            except Exception as e:
                self._send(500, {"error": str(e)})
        else:
            self._send(404, {"error": "Not found"})

    def _route(self):
        return self.path.split("?")[0].rstrip("/") or "/"

    def _send(self, status, payload):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def _send_html(self, status, html):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))
