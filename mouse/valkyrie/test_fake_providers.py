"""Offline check of run.py request/response handling against fake provider endpoints (no API spend)."""
import json, os, subprocess, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
seen = []
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        seen.append((self.path, dict(self.headers), body))
        txt = "reasoning...\nAnswer: B"
        if self.path.endswith("/v1/messages"): out = {"content": [{"type": "text", "text": txt}]}
        elif self.path.endswith("/chat/completions"): out = {"choices": [{"message": {"content": txt}}]}
        elif ":generateContent" in self.path: out = {"candidates": [{"content": {"parts": [{"text": "hmm", "thought": True}, {"text": txt}]}}]}
        else: self.send_response(404); self.end_headers(); return
        b = json.dumps(out).encode(); self.send_response(200); self.send_header("content-length", str(len(b))); self.end_headers(); self.wfile.write(b)
srv = HTTPServer(("127.0.0.1", 0), H); port = srv.server_port
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{port}"
env = {**os.environ, "ANTHROPIC_BASE_URL": base + "/passthrough/anthropic", "ANTHROPIC_API_KEY": "tok",
       "OPENAI_BASE_URL": base + "/passthrough/openai/v1", "OPENAI_API_KEY": "tok",
       "GOOGLE_GEMINI_BASE_URL": base + "/passthrough/google", "GEMINI_API_KEY": "tok",
       "OPENROUTER_BASE_URL": base + "/passthrough/openrouter/api/v1", "OPENROUTER_API_KEY": "tok"}
d = tempfile.mkdtemp(); prob = os.path.join(d, "p.md"); open(prob, "w").write("Q? A) x B) y")
ok = True
for m in ["anthropic/claude-x", "openai/gpt-x", "google/gemini-x", "openrouter/meta/llama-x"]:
    out = os.path.join(d, "a.txt")
    r = subprocess.run([sys.executable, "agents/oneshot-mcq/run.py", prob, "--model", m, "--max-tokens", "100", "--temperature", "", "--out", out], env=env, capture_output=True, text=True)
    txt = open(out).read() if os.path.exists(out) else ""
    path = seen[-1][0] if seen else None
    good = r.returncode == 0 and txt.strip().endswith("Answer: B") and "hmm" not in txt
    ok &= good
    print(("OK  " if good else "FAIL"), m, "->", path, "| model sent:", seen[-1][2].get("model") if seen and "model" in seen[-1][2] else "(in url)", r.stderr[-200:])
print("ALL OK" if ok else "FAILURES")

# Gateway mode without proxy: only MODEL_GATEWAY_URL + rotating token file
tokf = os.path.join(d, "gateway_token"); open(tokf, "w").write("rotated-tok\n")
genv = {k: v for k, v in os.environ.items()}
genv.update({"MODEL_GATEWAY_URL": base, "MODEL_GATEWAY_API_KEY": "stale", "MODEL_GATEWAY_API_KEY_FILE": tokf})
for m in ["anthropic/claude-x", "openai/gpt-x", "google/gemini-x"]:
    out = os.path.join(d, "g.txt")
    r = subprocess.run([sys.executable, "agents/oneshot-mcq/run.py", prob, "--model", m, "--out", out], env=genv, capture_output=True, text=True)
    path, hdr = seen[-1][0], {k.lower(): v for k, v in seen[-1][1].items()}
    key = hdr.get("x-api-key") or hdr.get("x-goog-api-key") or hdr.get("authorization", "")
    good = r.returncode == 0 and path.startswith("/passthrough/") and "rotated-tok" in key
    print(("OK  " if good else "FAIL"), "gateway", m, "->", path, r.stderr[-200:])
