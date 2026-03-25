#!/usr/bin/env python3
import json
import logging
import signal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = "127.0.0.1"
PORT = 8000
server_ref = {"server": None}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("local")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log.info("%s - %s", self.client_address[0], fmt % args)

    def _send_json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            return self._send_json(200, {"ok": True, "status": "healthy"})
        return self._send_json(404, {"ok": False, "error": "not_found", "path": self.path})

    def do_POST(self):
        if self.path.startswith("/echo"):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b""
            return self._send_json(200, {
                "ok": True,
                "method": "POST",
                "path": self.path,
                "body": body.decode("utf-8", errors="replace"),
            })
        return self._send_json(404, {"ok": False, "error": "not_found", "path": self.path})


def shutdown(signum=None, frame=None):
    log.info("Shutdown requested: signal=%s", signum)
    srv = server_ref["server"]
    if srv:
        srv.shutdown()
        srv.server_close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server_ref["server"] = server
    log.info("Local server listening on http://%s:%s", HOST, PORT)
    server.serve_forever()
