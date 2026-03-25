# relay-project

Minimal reverse HTTP relay for restricted sandbox environments.

## Features

- No ngrok / cloudflared / ssh required
- Works over plain outbound HTTP from sandbox to VPS
- Supports GET and POST
- Handles headers, query string, request body
- Secret-based agent auth
- Graceful shutdown
- Session cleanup

## Install

```bash
pip install -r requirements.txt
