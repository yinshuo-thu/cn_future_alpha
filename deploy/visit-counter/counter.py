#!/usr/bin/env python3
"""Tiny stdlib-only visit counter for https://autoalpha.cn/cn_future_alpha/.

Routes (behind nginx at /cn_future_alpha/visits):
  GET .../visits        -> peek  (returns current count, no increment)
  GET .../visits/hit    -> bump  (atomically +1, returns new count)
Returns JSON {"count": N}. Count persists in COUNT_FILE with flock.
"""
import os
import json
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST, PORT = "127.0.0.1", 8766
COUNT_FILE = "/opt/cnfa-counter/count.txt"


def bump(delta):
    fd = os.open(COUNT_FILE, os.O_RDWR | os.O_CREAT, 0o644)
    with os.fdopen(fd, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            raw = f.read().strip()
            n = int(raw) if raw.lstrip("-").isdigit() else 0
            if delta:
                n += delta
                f.seek(0)
                f.truncate()
                f.write(str(n))
                f.flush()
                os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return n


class Handler(BaseHTTPRequestHandler):
    def _json(self, n):
        body = json.dumps({"count": n}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        n = bump(1 if path.endswith("/hit") else 0)
        self._json(n)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
