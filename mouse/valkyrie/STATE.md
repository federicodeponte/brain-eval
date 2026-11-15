# Valkyrie integration: mouse-V1 MCQ benchmark (state 2026-11-15 ~03:40 CET)

## How custom benchmarks work on Valkyrie
- Two ways: (a) a **dataset**, a folder of Harbor-format task dirs (task.toml, instruction.md, tests/test.sh, optional solution/), with no service to host; (b) a **benchmark service** (create-benchmark-service, FastAPI) with load/retrieve/setup/evaluate/score hooks. We use (a): it is the fastest and needs no deploy.
- Every run has the agentic shape: each task runs in a sandbox (Modal/Daytona when hosted, Docker with a local Tracker). The agent gets instruction.md. The verifier `tests/test.sh` is uploaded only at grading time, so the answer key never reaches the agent, and it writes `/logs/verifier/reward.txt`. Run score = mean reward x 100 (no metric.py needed).
- A plain prompt -> answer benchmark only needs an agent that makes one model call. We wrote `oneshot-mcq` (stdlib Python, no install step): it reads instruction.md, sends it as one user message, and writes the reply to /app/answer.txt. Registry agents (claude_code, codex, terminus2, ...) also run on the same dataset unchanged.
- Models: `--model provider/name` (anthropic/..., openai/..., google/..., openrouter/...). On hosted Valkyrie, a **project** created with `--budget` gets a Vals Model Gateway key. The tracker then injects either ANTHROPIC_/OPENAI_/GOOGLE_GEMINI_/OPENROUTER_ BASE_URL+API_KEY (egress-proxy mode) or MODEL_GATEWAY_URL plus a rotating token file. oneshot-mcq handles both. The alternative is our own keys via `valkyrie secret set` + `-s ENV_VAR secret`.
- Results: `valkyrie run fetch/list` shows the Final score. `valkyrie run results RUN --path x.json` gives per-task rewards and verifier output. `leaderboard.py results-*.json --md` builds the model table. `valkyrie tui` lets you browse runs.

## What is built (AX41 ~/work/vals-mouse/valkyrie/)
- `build_dataset.py tasks.jsonl OUT` -> Harbor task folders (image python:3.12-slim, workdir /app, agent timeout 1800s). It appends "last line `Answer: X`" to each prompt and writes solution/solve.sh (the oracle).
- `grade.py`: exact-match grader, copied into each task's tests/. It takes the last `Answer: X` line or a bare single letter, accepts only the first k letters, and scores an unparseable reply 0. It writes reward.txt + details.json.
- `dataset/`: built from ../tasks.jsonl (90 tasks: 60 with k=4, 30 with k=8). `dataset-dummy/`: 3 dummy tasks.
- `agents/oneshot-mcq/`: contract.yaml + run.py (anthropic/openai/google/openrouter, plus `mock/X` for free testing; kwargs max_tokens=16000, temperature).
- `leaderboard.py`, `test_fake_providers.py` (offline request-shape test), and `local_tracker.sh` + `compose.override.yml` + `local-server.yaml` (local Tracker on ports 8031/8035/55432, because 8000/8002/5432 are taken on AX41).
- CLI: `~/.local/bin/valkyrie` (uv tool install valkyrie-cli). A newer uv lives in ~/work/vals-mouse/bin, because system uv 0.10 is too old for the tracker.

## Tested (no login, no model spend)
- All 90 tasks pass the tracker's own `validate_task` (0 errors, 0 warnings).
- Local Tracker + Docker sandboxes, end to end, `--dataset ./dataset`: oracle = 100.0% (90/90). oneshot-mcq with mock/A = 21.1% (19/90, exactly the number of A answers). Dummy set: oracle 100%, mock/A 33.3%, mock/G 33.3% (G is not a valid letter when k=4, so it is unparsed and scores 0, as designed).
- Grader: 10 parse cases pass. Agent: the fake-server test passes for all 4 providers, plus the gateway fallback with a rotating token file.
- Local tracker is still running (stop it with `pkill -f "tracker.serve --config"` and `docker compose -f Valkyrie/services/tracker/docker-compose.yml -f compose.override.yml down`).
- Gotcha: the first `--agent ./agents/oneshot-mcq` publishes the agent. After editing it, run `valkyrie agent push agents/oneshot-mcq` and then use `--agent oneshot-mcq`.

## Commands for Federico (on AX41: cd ~/work/vals-mouse/valkyrie; unset VALKYRIE_URL)
```
valkyrie login                      # browser, Google
valkyrie whoami; valkyrie org list  # if the hackathon org is separate: valkyrie org use <org>
valkyrie provider list              # the org needs modal or daytona configured; if not, ask Vals staff
valkyrie project create mouse-v1 --name "Mouse V1 MCQ" --spend-type development \
  --subcategory new_benchmark_generation --budget 50 \
  --budget-reason "3 models x 90 single-call tasks, about 1M tokens"
valkyrie agent push agents/oneshot-mcq
valkyrie dataset push mouse-v1/mcq ./dataset --tag v1
# smoke test: 2 tasks, one model
valkyrie run start --agent oneshot-mcq --dataset mouse-v1/mcq@v1 --project mouse-v1 \
  --model anthropic/claude-sonnet-4-6 --slice 0:2 --concurrency 2
# full runs (swap the model ids for the ones the gateway offers)
for M in anthropic/claude-sonnet-4-6 openai/gpt-5 google/gemini-2.5-pro; do
  valkyrie run start --agent oneshot-mcq --dataset mouse-v1/mcq@v1 --project mouse-v1 \
    --model $M --concurrency 10 -l mouse-v1
done
valkyrie run list                   # then, for each run:
valkyrie run results <RUN_ID> --path results-<model>.json
python3 leaderboard.py results-*.json --md
```
Add `--provider modal` (or daytona) if the org has no default provider.

## Blockers / unknowns
- Hosted sandboxes need the org to have Modal or Daytona connected (`valkyrie provider list`). We do not know whether the hackathon org has one; ask Vals staff.
- We do not know whether projects in this org get a Gateway key (the help text says the budget flags are "required where projects get a Gateway key"), or which model ids the gateway allows. Without a gateway: `valkyrie secret set anthropic-api-key` etc. and `-s ANTHROPIC_API_KEY anthropic-api-key` on run start. That spends our own money and needs Federico's OK.
- The project spend type/subcategory (development / new_benchmark_generation) is a guess.
- The prompts already end with "Answer with a single letter."; we append the `Answer: X` rule. The grader accepts both forms.
- Approximate size: prompts are 11-21k characters (roughly 3-6k tokens) each, so one model over 90 tasks is roughly 0.5M input tokens plus the output.

## Human neurons dataset (added 2026-11-14)
- dataset-human/: 40 tasks from ~/work/vals-hyp/human-neurons/tasks.jsonl (k=4 added), same builder.
- Push: valkyrie dataset push human-v1/mcq ./dataset-human --tag v1
