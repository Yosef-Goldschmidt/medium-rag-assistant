"""Single Vercel Python entrypoint. Routes by URL path:
    GET  /api/stats   -> RAG hyperparameters
    POST /api/prompt  -> answer a question with the RAG pipeline

Imports of the heavy RAG chain are done lazily inside the handlers so that
/api/stats (which only needs config) works even if the model/DB clients aren't
reachable.
"""
import os
import sys
import json
from http.server import BaseHTTPRequestHandler

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self._route() == "/api/stats":
            import config
            self._send(200, {
                "chunk_size": config.CHUNK_SIZE,
                "overlap_ratio": config.OVERLAP_RATIO,
                "top_k": config.TOP_K,
            })
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
