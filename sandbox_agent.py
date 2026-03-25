#!/usr/bin/env python3
import base64
import json
import logging
import os
import signal
import time

import requests

RELAY_BASE = os.environ.get("RELAY_BASE", "http://149.62.209.88:9000")
SECRET = os.environ.get("RELAY_SECRET", "CHANGE_ME_SECRET")
LOCAL_BASE = os.environ.get("LOCAL_BASE", "http://127.0.0.1:8000")
POLL_TIMEOUT = int(os.environ.get("AGENT_POLL_TIMEOUT", "35"))
HTTP_TIMEOUT = int(os.environ.get("AGENT_HTTP_TIMEOUT", "20"))
RETRY_DELAY = int(os.environ.get("AGENT_RETRY_DELAY", "2"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("agent")

RUNNING = True
TOKEN = None
PUBLIC_BASE = None


def shutdown(signum=None, frame=None):
    global RUNNING
    log.info("Shutdown requested: signal=%s", signum)
    RUNNING = False


def register():
    headers = {"X-SECRET": SECRET}
    r = requests.post(f"{RELAY_BASE}/register", headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(f"register failed: {data}")
    return data["token"], data["public_base"]


def poll_once():
    headers = {
        "X-SECRET": SECRET,
        "X-TOKEN": TOKEN,
    }
    r = requests.post(f"{RELAY_BASE}/poll", headers=headers, timeout=POLL_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data.get("request")


def respond(request_id: str, response: dict):
    payload = {
        "token": TOKEN,
        "request_id": request_id,
        "response": response,
    }
    headers = {"X-SECRET": SECRET}
    r = requests.post(f"{RELAY_BASE}/respond", headers=headers, json=payload, timeout=15)
    r.raise_for_status()


def proxy_request(req: dict):
    req_id = req["id"]
    method = req.get("method", "GET").upper()
    path = req.get("path", "/")
    query = req.get("query", "")
    headers = req.get("headers", {}).copy()
    body_b64 = req.get("body_b64", "")

    url = f"{LOCAL_BASE}{path}"
    if query:
        url += f"?{query}"

    headers.pop("Host", None)
    headers["X-Forwarded-By"] = "sandbox-agent"

    body = base64.b64decode(body_b64) if body_b64 else b""

    try:
        resp = requests.request(
            method=method,
            url=url,
            headers=headers,
            data=body if body else None,
            timeout=HTTP_TIMEOUT,
        )
        response_payload = {
            "status": resp.status_code,
            "headers": {
                "Content-Type": resp.headers.get("Content-Type", "application/octet-stream")
            },
            "body_b64": base64.b64encode(resp.content).decode("ascii"),
        }
    except Exception as e:
        response_payload = {
            "status": 502,
            "headers": {"Content-Type": "application/json; charset=utf-8"},
            "body_b64": base64.b64encode(
                json.dumps({"ok": False, "error": "local_request_failed", "detail": str(e)}).encode("utf-8")
            ).decode("ascii"),
        }

    respond(req_id, response_payload)


def local_self_test():
    r = requests.get(f"{LOCAL_BASE}/health", timeout=5)
    r.raise_for_status()
    return r.text


if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    log.info("Checking local service: %s/health", LOCAL_BASE)
    try:
        body = local_self_test()
        log.info("Local self-test OK: %s", body[:200])
    except Exception as e:
        log.error("Local self-test failed: %s", e)
        raise SystemExit(1)

    TOKEN, PUBLIC_BASE = register()
    log.info("Registered token: %s...", TOKEN[:8])
    log.info("Public base: %s", PUBLIC_BASE)
    log.info("Public health: %s/health", PUBLIC_BASE)

    while RUNNING:
        try:
            req = poll_once()
            if req:
                log.info("Request %s %s", req.get("method"), req.get("path"))
                proxy_request(req)
        except Exception as e:
            log.warning("Poll/proxy error: %s", e)
            time.sleep(RETRY_DELAY)

    log.info("Agent stopped")
