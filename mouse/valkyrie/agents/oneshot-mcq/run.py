#!/usr/bin/env python3
"""One-shot model call for Valkyrie (stdlib only).

Reads the problem statement, sends it as one user message to provider/model, writes the reply
text to --out and the raw API response to /logs/agent/response.json.

Base URLs and keys come from the env vars the Vals Model Gateway injects into the sandbox:
  OPENAI_BASE_URL/OPENAI_API_KEY, ANTHROPIC_BASE_URL/ANTHROPIC_API_KEY,
  GOOGLE_GEMINI_BASE_URL/GEMINI_API_KEY (or GOOGLE_API_KEY), OPENROUTER_BASE_URL/OPENROUTER_API_KEY.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

LOG_DIR = Path(os.environ.get("ONESHOT_LOG_DIR", "/logs/agent"))


def env(*names, default=None):
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


GATEWAY_ROUTES = {
    "anthropic": "/passthrough/anthropic",
    "openai": "/passthrough/openai/v1",
    "google": "/passthrough/google",
    "openrouter": "/passthrough/openrouter/api/v1",
}


def endpoint(provider, base_var, key_vars, default_base):
    """Provider base URL + key. Order: provider env vars (set by the egress-proxy gateway mode or -s
    secrets), then MODEL_GATEWAY_URL + scoped token (read fresh from the rotating token file)."""
    base, key = env(base_var), env(*key_vars)
    gw = os.environ.get("MODEL_GATEWAY_URL")
    if not base and gw:
        base = gw.rstrip("/") + GATEWAY_ROUTES[provider]
        tok_file = os.environ.get("MODEL_GATEWAY_API_KEY_FILE")
        if tok_file and Path(tok_file).exists():
            key = Path(tok_file).read_text().strip()
        else:
            key = os.environ.get("MODEL_GATEWAY_API_KEY", key)
    return (base or default_base).rstrip("/"), key or ""


def post(url, headers, body, attempts=6):
    data = json.dumps(body).encode()
    delay = 5.0
    for i in range(attempts):
        req = urllib.request.Request(url, data=data, headers={"content-type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:500]
            if e.code in (408, 409, 429, 500, 502, 503, 504, 529) and i < attempts - 1:
                print(f"HTTP {e.code}, retry in {delay:.0f}s: {msg}", file=sys.stderr)
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            raise RuntimeError(f"HTTP {e.code}: {msg}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            if i < attempts - 1:
                print(f"network error, retry in {delay:.0f}s: {e}", file=sys.stderr)
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            raise


def call(provider, model, prompt, max_tokens, temperature):
    if provider == "mock":
        return {"mock": True}, f"Mock reply.\nAnswer: {model.strip()[:1].upper() or 'A'}"

    if provider == "anthropic":
        base, key = endpoint("anthropic", "ANTHROPIC_BASE_URL", ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
                             "https://api.anthropic.com")
        body = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
        if temperature is not None:
            body["temperature"] = temperature
        raw = post(f"{base}/v1/messages",
                   {"x-api-key": key,
                    "anthropic-version": "2023-06-01"}, body)
        text = "".join(b.get("text", "") for b in raw.get("content", []) if b.get("type") == "text")
        return raw, text

    if provider in ("openai", "openrouter"):
        if provider == "openai":
            base, key = endpoint("openai", "OPENAI_BASE_URL", ("OPENAI_API_KEY",), "https://api.openai.com/v1")
        else:
            base, key = endpoint("openrouter", "OPENROUTER_BASE_URL", ("OPENROUTER_API_KEY",),
                                 "https://openrouter.ai/api/v1")
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": max_tokens}
        if temperature is not None:
            body["temperature"] = temperature
        raw = post(f"{base}/chat/completions", {"authorization": f"Bearer {key}"}, body)
        text = raw["choices"][0]["message"].get("content") or ""
        return raw, text

    if provider in ("google", "gemini"):
        base, key = endpoint("google", "GOOGLE_GEMINI_BASE_URL", ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
                             "https://generativelanguage.googleapis.com")
        cfg = {"maxOutputTokens": max_tokens}
        if temperature is not None:
            cfg["temperature"] = temperature
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": cfg}
        raw = post(f"{base}/v1beta/models/{model}:generateContent",
                   {"x-goog-api-key": key}, body)
        parts = (raw.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        return raw, text

    raise SystemExit(f"unknown provider {provider!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", default="16000")
    ap.add_argument("--temperature", default="")
    ap.add_argument("--out", default="/app/answer.txt")
    a = ap.parse_args()

    provider, _, model = a.model.partition("/")
    max_tokens = int(a.max_tokens) if a.max_tokens.strip().isdigit() else 16000
    try:
        temperature = float(a.temperature)
    except ValueError:
        temperature = None

    prompt = Path(a.problem).read_text()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        raw, text = call(provider, model, prompt, max_tokens, temperature)
    except Exception as e:
        err = os.environ.get("VALKYRIE_ERROR_PATH")
        if err:
            Path(err).write_text(f"ModelCallError: {str(e)[:300]}\n")
        raise
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text)
    (LOG_DIR / "response.json").write_text(json.dumps(
        {"model": a.model, "seconds": round(time.time() - t0, 2), "text": text, "raw": raw}, indent=1))
    print(text[-400:])


if __name__ == "__main__":
    main()
