
"""POST /api/prompt  ->  answer a question with the RAG pipeline."""
import os
import sys
import json
from http.server import BaseHTTPRequestHandler

# Make the src/ modules importable (they use bare imports among themselves).
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from rag import answer_question

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw or b"{}")
            question = body.get("question", "")
            if not question:
                self._send(400, {"error": "Missing 'question' in request body."})
                return
            self._send(200, answer_question(question))
        except Exception as e:
            self._send(500, {"error": str(e)})

    def _send(self, status, payload):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))
