#!/usr/bin/env python3
"""Unique-IP visit counter for https://autoalpha.cn/cn_future_alpha/.

Counts each client IP at most once (ever). The real client IP comes from the
nginx-set X-Real-IP header; IPs are stored hashed (sha256, salted) so no raw
addresses are kept on disk. The count is the number of distinct IPs.

Routes (behind nginx at /cn_future_alpha/visits):
  GET .../visits        -> peek (current unique count, no change)
  GET .../visits/hit    -> register this IP if new; returns the unique count
Response: {"count": N}.

Files (in /opt/cnfa-counter): ips.txt = one hashed IP per line (the unique set);
count.txt mirrors len(set) for easy inspection.
"""
import os
import json
import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST, PORT = "127.0.0.1", 8766
DIR = "/opt/cnfa-counter"
IPS_FILE = os.path.join(DIR, "ips.txt")
COUNT_FILE = os.path.join(DIR, "count.txt")
SALT = "cnfa-visit-v1"

_lock = threading.Lock()
_seen = set()


def _load():
    try:
        with open(IPS_FILE, encoding="utf-8") as f:
            for line in f:
                h = line.strip()
                if h:
                    _seen.add(h)
    except FileNotFoundError:
        pass


def _client_ip(handler):
    ip = (handler.headers.get("X-Real-IP") or "").strip()
    if not ip:
        xff = handler.headers.get("X-Forwarded-For", "")
        ip = xff.split(",")[0].strip() if xff else ""
    if not ip:
        ip = handler.client_address[0]
    return ip


def _hash(ip):
    return hashlib.sha256((SALT + "|" + ip).encode("utf-8")).hexdigest()[:16]


def register(ip):
    """Add the IP if unseen; return the current unique count."""
    h = _hash(ip)
    with _lock:
        if h not in _seen:
            _seen.add(h)
            with open(IPS_FILE, "a", encoding="utf-8") as f:
                f.write(h + "\n")
                f.flush()
                os.fsync(f.fileno())
            with open(COUNT_FILE, "w", encoding="utf-8") as f:
                f.write(str(len(_seen)))
                f.flush()
                os.fsync(f.fileno())
        return len(_seen)


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
        if path.endswith("/hit"):
            n = register(_client_ip(self))
        else:
            with _lock:
                n = len(_seen)
        self._json(n)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    _load()
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
