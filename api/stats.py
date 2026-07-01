"""GET /api/stats  ->  the current RAG hyperparameters."""
import os
import sys
import json
from http.server import BaseHTTPRequestHandler

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import config


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        stats = {
            "chunk_size": config.CHUNK_SIZE,
            "overlap_ratio": config.OVERLAP_RATIO,
            "top_k": config.TOP_K,
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(stats).encode("utf-8"))
