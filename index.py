"""Single Vercel Python entrypoint. Routes by URL path:
    GET  /            -> chat GUI (HTML)
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
<title>Yosef's RAG Assistant</title>
<style>
  :root {
    --bg:#f6f6f3; --card:#ffffff; --fg:#1b1b1f; --muted:#6b7280;
    --line:#ececea; --accent:#0f9d6e; --accent-dark:#0b7d58; --ring:rgba(15,157,110,.15);
  }
  * { box-sizing: border-box; }
  html, body { margin:0; }
  body {
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,sans-serif;
    color:var(--fg); background:
      radial-gradient(900px 400px at 50% -120px, #e7f6ef 0%, rgba(231,246,239,0) 70%),
      var(--bg);
    line-height:1.55; padding:0 20px 90px;
  }
  .wrap { max-width:760px; margin:0 auto; }

  /* Header */
  header { text-align:center; padding:52px 0 28px; }
  .monogram {
    width:56px; height:56px; border-radius:16px; margin:0 auto 16px;
    display:grid; place-items:center; color:#fff; font-weight:700; font-size:1.4rem;
    background:linear-gradient(135deg,var(--accent),var(--accent-dark));
    box-shadow:0 8px 24px rgba(15,157,110,.28);
    font-family:Georgia,"Times New Roman",serif;
  }
  h1 { font-family:Georgia,"Times New Roman",serif; font-size:2rem; margin:0 0 8px; letter-spacing:-.01em; }
  .tagline { color:var(--muted); margin:0 auto; max-width:560px; }
  .badges { display:flex; gap:8px; justify-content:center; flex-wrap:wrap; margin-top:16px; }
  .badge { font-size:.75rem; color:var(--accent-dark); background:#e9f7f0;
           border:1px solid #d3efe1; padding:4px 10px; border-radius:999px; }

  /* How it works */
  .how { display:flex; gap:12px; justify-content:center; flex-wrap:wrap; margin:22px 0 8px; }
  .step { font-size:.82rem; color:var(--muted); background:var(--card); border:1px solid var(--line);
          padding:8px 12px; border-radius:10px; }
  .step b { color:var(--fg); }

  /* Card */
  .card { background:var(--card); border:1px solid var(--line); border-radius:16px;
          padding:20px; box-shadow:0 1px 2px rgba(0,0,0,.03); margin-top:20px; }

  .examples { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:14px; }
  .chip { font-size:.8rem; border:1px solid var(--line); background:#fbfbfa; color:#374151;
          padding:6px 11px; border-radius:999px; cursor:pointer; transition:.15s; }
  .chip:hover { border-color:var(--accent); color:var(--accent-dark); background:#f2fbf7; }

  textarea { width:100%; min-height:80px; padding:13px 14px; border:1px solid var(--line);
             border-radius:12px; font:inherit; resize:vertical; background:#fcfcfb; transition:.15s; }
  textarea:focus { outline:none; border-color:var(--accent); box-shadow:0 0 0 4px var(--ring); }

  .row { display:flex; gap:10px; align-items:center; margin-top:12px; flex-wrap:wrap; }
  button { border:0; padding:11px 20px; border-radius:10px; font:inherit; font-weight:600;
           cursor:pointer; transition:.15s; }
  .primary { background:var(--accent); color:#fff; box-shadow:0 6px 16px rgba(15,157,110,.25); }
  .primary:hover { background:var(--accent-dark); }
  .secondary { background:#fff; color:#374151; border:1px solid var(--line); }
  .secondary:hover { border-color:var(--accent); color:var(--accent-dark); }
  button:disabled { opacity:.5; cursor:default; box-shadow:none; }
  #stats { margin-left:auto; color:var(--muted); font-size:.86rem; }

  .spinner { display:inline-block; width:15px; height:15px; border:2px solid #cbd5d1;
             border-top-color:var(--accent); border-radius:50%; animation:spin .8s linear infinite;
             vertical-align:-2px; margin-right:8px; }
  @keyframes spin { to { transform:rotate(360deg); } }

  #answerCard { display:none; }
  #answer { white-space:pre-wrap; }
  .label { font-size:.72rem; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); margin-bottom:8px; }

  details { margin-top:14px; }
  summary { cursor:pointer; color:var(--accent-dark); font-weight:600; font-size:.9rem; }
  summary:hover { text-decoration:underline; }
  .chunk { border-top:1px solid var(--line); padding:11px 0; font-size:.88rem; }
  .chunk:first-child { border-top:0; }
  .chunk .meta { color:var(--muted); font-size:.78rem; margin-bottom:5px; }
  .score { color:var(--accent-dark); font-weight:600; }
  pre { white-space:pre-wrap; background:#fbfbfa; border:1px solid var(--line);
        border-radius:10px; padding:12px; font-size:.8rem; overflow-x:auto; }

  footer { text-align:center; color:var(--muted); font-size:.82rem; margin-top:36px; }
  footer a { color:var(--accent-dark); text-decoration:none; }
  footer a:hover { text-decoration:underline; }
</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="monogram">Y</div>
    <h1>Yosef's RAG Assistant</h1>
    <p class="tagline">Ask anything about a library of Medium articles. The assistant retrieves the
      most relevant passages and answers <b>only</b> from them — never from outside knowledge. If the
      articles don't cover your question, it will say so.</p>
    <div class="badges">
      <span class="badge">~7,600 articles</span>
      <span class="badge">29k passages</span>
      <span class="badge">Pinecone + gpt-5-mini</span>
    </div>
    <div class="how">
      <span class="step">🔎 <b>Retrieve</b> relevant passages</span>
      <span class="step">🧩 <b>Augment</b> the prompt</span>
      <span class="step">✍️ <b>Answer</b> from context</span>
    </div>
  </header>

  <div class="card">
    <div class="examples" id="examples"></div>
    <textarea id="q" placeholder="e.g. Find an article about smell training and the brain. Give the title and author."></textarea>
    <div class="row">
      <button class="primary" id="ask">Ask</button>
      <button class="secondary" id="statsBtn">Show config</button>
      <span id="stats"></span>
    </div>
  </div>

  <div class="card" id="answerCard">
    <div class="label">Answer</div>
    <div id="answer"></div>

    <details id="ctxWrap" style="display:none">
      <summary>Retrieved context</summary>
      <div id="context"></div>
    </details>

    <details id="promptWrap" style="display:none">
      <summary>Augmented prompt (sent to the model)</summary>
      <div class="label" style="margin-top:12px">System</div><pre id="sysPrompt"></pre>
      <div class="label">User</div><pre id="userPrompt"></pre>
    </details>
  </div>

  <footer>
    Built by Yosef Goldschmidt ·
    <a href="https://github.com/Yosef-Goldschmidt/medium-rag-assistant" target="_blank">source on GitHub</a>
  </footer>

</div>

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
  c.onclick = () => { document.getElementById("q").value = text; document.getElementById("q").focus(); };
  examplesEl.appendChild(c);
});

const askBtn = document.getElementById("ask");
const statsBtn = document.getElementById("statsBtn");
const answerCard = document.getElementById("answerCard");
const answerEl = document.getElementById("answer");
const statsEl = document.getElementById("stats");
const ctxWrap = document.getElementById("ctxWrap");
const contextEl = document.getElementById("context");
const promptWrap = document.getElementById("promptWrap");

async function ask() {
  const question = document.getElementById("q").value.trim();
  if (!question) return;
  askBtn.disabled = true;
  answerCard.style.display = "block";
  answerEl.innerHTML = '<span class="spinner"></span><span style="color:var(--muted)">Thinking… (this can take 10–30s)</span>';
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
        d.innerHTML = '<div class="meta"><b>' + (c.title || "") + "</b>  ·  " +
          '<span class="score">score ' + (c.score != null ? c.score.toFixed(3) : "?") + "</span>" +
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
    statsEl.textContent = "chunk_size " + data.chunk_size +
      " · overlap " + data.overlap_ratio + " · top_k " + data.top_k;
  } catch (e) {
    statsEl.textContent = "Failed: " + e;
  } finally {
    statsBtn.disabled = false;
  }
}

askBtn.onclick = ask;
statsBtn.onclick = showStats;
document.getElementById("q").addEventListener("keydown", e => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") ask();
});
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
