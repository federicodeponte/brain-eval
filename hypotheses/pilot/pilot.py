#!/usr/bin/env python3
"""Guarded pilot: N items per hypothesis, 1 model, 1 worker, via claude -p. Uses AX41 SpendGuard."""
import sys, json, re, argparse, os
from pathlib import Path
sys.path.insert(0, '/home/federicodeponte/sbs-20261023')
from spend_guard import SpendGuard, GuardError, add_arguments

BASE = Path.home() / 'work/vals-hyp/pilot'
SRC = {'mouse': Path.home()/'work/vals-mouse/tasks.jsonl', 'mouse8': Path.home()/'work/vals-mouse/tasks.jsonl'}
for d in (Path.home()/'work/vals-hyp').iterdir():
    if (d/'tasks.jsonl').exists() and d.name not in ('smell',):
        SRC[d.name] = d/'tasks.jsonl'

def parse(txt):
    m = re.search(r'(?i)answer\s*[:is]*\s*\**\s*\(?([A-H])\b', txt)
    if m: return m.group(1).upper()
    first = txt.strip().splitlines()[0] if txt.strip() else ''
    m = re.search(r'\b([A-H])\b', first)
    if m: return m.group(1)
    m = re.findall(r'\b([A-H])\b', txt)
    return m[0] if m else None

p = argparse.ArgumentParser(); add_arguments(p)
p.add_argument('--hyp', nargs='+', required=True); p.add_argument('--n', type=int, default=10)
p.add_argument('--model', default='opus')
a = p.parse_args()
g = SpendGuard(BASE, a)
out = BASE / 'results.jsonl'
empty = BASE / 'empty'; empty.mkdir(parents=True, exist_ok=True)
for h in a.hyp:
    items = [json.loads(l) for l in open(SRC[h])]
    if h == 'mouse': items = [i for i in items if i.get('k') == 4]
    if h == 'mouse8': items = [i for i in items if i.get('k') == 8]
    for it in items[:a.n]:
        key = f'judge:{h}:{it["id"]}:{a.model}'
        if g.attempted(key): continue
        try:
            with g.claim(key) as entry:
                cmd = ['claude', '-p', '--model', a.model, '--no-session-persistence',
                       '--disallowedTools', 'Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,Task,NotebookEdit'] + g.claude_flags()
                r = g.run_process(cmd, input=it['prompt'], cwd=empty)
                try:
                    j = json.loads(r.stdout)
                except Exception:
                    j = {'result': '', 'raw': r.stdout[-500:], 'err': r.stderr[-500:]}
                entry['provider'] = 'claude'
                entry['reported_cost_usd'] = j.get('total_cost_usd')
                txt = (j.get('result') or '').strip()
                pred = parse(txt)
                rec = {'hyp': h, 'id': it['id'], 'model': a.model, 'answer': it['answer'], 'pred': pred,
                       'correct': pred == it['answer'], 'cost': j.get('total_cost_usd'), 'text': txt[:3000]}
                with open(out, 'a') as f: f.write(json.dumps(rec) + '\n')
                print(h, it['id'], it['answer'], pred, rec['correct'], flush=True)
        except GuardError as e:
            print('GUARD STOP:', e, flush=True); sys.exit(2)
