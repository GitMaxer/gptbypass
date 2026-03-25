#!/usr/bin/env python3
import base64
import json
import logging
import os
import queue
import secrets
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HOST = os.environ.get("RELAY_HOST", "0.0.0.0")
PORT = int(os.environ.get("RELAY_PORT", "9000"))
SECRET = os.environ.get("RELAY_SECRET", "CHANGE_ME_SECRET")
POLL_TIMEOUT = int(os.environ.get("RELAY_POLL_TIMEOUT", "25"))
RESULT_TIMEOUT = int(os.environ.get("RELAY_RESULT_TIMEOUT", "30"))
SESSION_TTL = int(os.environ.get("RELAY_SESSION_TTL", "3600"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("relay")

sessions = {}
sessions_lock = threading.Lock()
server_ref = {"server": None}


def now() -> float:
    return time.time()


class Session:
    def __init__(self, token: str):
        self.token = token
        self.pending = queue.Queue()
        self.results = {}
        self.cond = threading.Condition()
        self.last_seen = now()

    def touch(self):
        self.last_seen = now()


def get_session(token: str):
    with sessions_lock:
        session = sessions.get(token)
        if session:
            session.touch()
        return session


def create_session() -> str:
    token = secrets.token_urlsafe(24)
    with sessions_lock:
        sessions[token] = Session(token)
    return token


def cleanup_sessions():
    while True:
        cutoff = now() - SESSION_TTL
        with sessions_lock:
            expired = [k for k, v in sessions.items() if v.last_seen < cutoff]
            for k in expired:
                del sessions[k]
        if expired:
            log.info("Cleaned %d expired session(s)", len(expired))
        time.sleep(300)


class Handler(BaseHTTPRequestHandler):
    server_version = "ReverseRelay/2.0"

    def log_message(self, fmt, *args):
        log.info("%s - %s", self.client_address[0], fmt % args)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(length) if length > 0 else b""

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth_ok(self) -> bool:
        return self.headers.get("X-SECRET") == SECRET

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/register":
            if not self._auth_ok():
                return self._json(403, {"ok": False, "error": "forbidden"})
            token = create_session()
            public_base = f"http://{self.headers.get('Host')}/t/{token}"
            log.info("Registered new agent token=%s", token[:8] + "...")
            return self._json(200, {"ok": True, "token": token, "public_base": public_base})

        if parsed.path == "/poll":
            if not self._auth_ok():
                return self._json(403, {"ok": False, "error": "forbidden"})
            token = self.headers.get("X-TOKEN") or ""
            session = get_session(token)
            if not session:
                return self._json(404, {"ok": False, "error": "unknown_token"})
            try:
                item = session.pending.get(timeout=POLL_TIMEOUT)
                session.touch()
                return self._json(200, {"ok": True, "request": item})
            except queue.Empty:
                return self._json(200, {"ok": True, "request": None})

        if parsed.path == "/respond":
            if not self._auth_ok():
                return self._json(403, {"ok": False, "error": "forbidden"})
            try:
                data = json.loads(self._read_body().decode("utf-8"))
            except Exception:
                return self._json(400, {"ok": False, "error": "bad_json"})

            token = data.get("token")
            request_id = data.get("request_id")
            response = data.get("response")

            if not token or not request_id or response is None:
                return self._json(400, {"ok": False, "error": "missing_fields"})

            session = get_session(token)
            if not session:
                return self._json(404, {"ok": False, "error": "unknown_token"})

            with session.cond:
                session.results[request_id] = response
                session.cond.notify_all()

            return self._json(200, {"ok": True})

        return self._json(404, {"ok": False, "error": "not_found"})

    def _handle_public_request(self):
        parsed = urlparse(self.path)

        if parsed.path == "/health":
            return self._json(200, {"ok": True, "role": "relay"})

        parts = parsed.path.split("/")
        if len(parts) < 4 or parts[1] != "t":
            return self._json(404, {"ok": False, "error": "not_found"})

        token = parts[2]
        session = get_session(token)
        if not session:
            return self._json(404, {"ok": False, "error": "unknown_token"})

        downstream_path = "/" + "/".join(parts[3:])
        request_id = secrets.token_urlsafe(12)

        raw_body = self._read_body()
        body_b64 = base64.b64encode(raw_body).decode("ascii") if raw_body else ""

        headers = {k: v for k, v in self.headers.items()}

        payload = {
            "id": request_id,
            "method": self.command,
            "path": downstream_path,
            "query": parsed.query,
            "headers": headers,
            "body_b64": body_b64,
            "created_at": now(),
        }

        session.pending.put(payload)
        session.touch()

        with session.cond:
            deadline = now() + RESULT_TIMEOUT
            while now() < deadline:
                if request_id in session.results:
                    response = session.results.pop(request_id)
                    status = int(response.get("status", 502))
                    resp_headers = response.get("headers", {}) or {}
                    resp_body_b64 = response.get("body_b64", "")
                    resp_body = base64.b64decode(resp_body_b64) if resp_body_b64 else b""

                    self.send_response(status)
                    for hk, hv in resp_headers.items():
                        if hk.lower() in {"content-length", "connection", "transfer-encoding"}:
                            continue
                        self.send_header(hk, hv)
                    self.send_header("Content-Length", str(len(resp_body)))
                    self.end_headers()
                    self.wfile.write(resp_body)
                    return

                session.cond.wait(timeout=1)

        return self._json(504, {"ok": False, "error": "agent_timeout"})

    def do_GET(self):
        self._handle_public_request()

    def do_POST_public(self):
        self._handle_public_request()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path in {"/register", "/poll", "/respond"}:
            return Handler.__dict__["_do_post_internal"](self)
        return self._handle_public_request()

    def _do_post_internal(self):
        parsed = urlparse(self.path)

        if parsed.path == "/register":
            if not self._auth_ok():
                return self._json(403, {"ok": False, "error": "forbidden"})
            token = create_session()
            public_base = f"http://{self.headers.get('Host')}/t/{token}"
            log.info("Registered new agent token=%s", token[:8] + "...")
            return self._json(200, {"ok": True, "token": token, "public_base": public_base})

        if parsed.path == "/poll":
            if not self._auth_ok():
                return self._json(403, {"ok": False, "error": "forbidden"})
            token = self.headers.get("X-TOKEN") or ""
            session = get_session(token)
            if not session:
                return self._json(404, {"ok": False, "error": "unknown_token"})
            try:
                item = session.pending.get(timeout=POLL_TIMEOUT)
                session.touch()
                return self._json(200, {"ok": True, "request": item})
            except queue.Empty:
                return self._json(200, {"ok": True, "request": None})

        if parsed.path == "/respond":
            if not self._auth_ok():
                return self._json(403, {"ok": False, "error": "forbidden"})
            try:
                data = json.loads(self._read_body().decode("utf-8"))
            except Exception:
                return self._json(400, {"ok": False, "error": "bad_json"})

            token = data.get("token")
            request_id = data.get("request_id")
            response = data.get("response")

            if not token or not request_id or response is None:
                return self._json(400, {"ok": False, "error": "missing_fields"})

            session = get_session(token)
            if not session:
                return self._json(404, {"ok": False, "error": "unknown_token"})

            with session.cond:
                session.results[request_id] = response
                session.cond.notify_all()

            return self._json(200, {"ok": True})

        return self._json(404, {"ok": False, "error": "not_found"})


def shutdown(signum=None, frame=None):
    log.info("Shutdown requested: signal=%s", signum)
    srv = server_ref["server"]
    if srv:
        srv.shutdown()
        srv.server_close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    cleaner = threading.Thread(target=cleanup_sessions, daemon=True)
    cleaner.start()

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server_ref["server"] = server
    log.info("Relay listening on http://%s:%s", HOST, PORT)
    server.serve_forever()
