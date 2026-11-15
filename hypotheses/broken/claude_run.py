#!/usr/bin/env python3
"""Broken-instrument v3 tasks via Claude Code (claude -p, subscription) on AX41, graded by the same tests/grade.py.

Per task: copy only agent-visible files (environment/data/*) into a fresh /tmp workdir, rewrite /app/... paths in
the instruction text to that workdir, run claude inside bwrap with /home hidden (only ~/.claude auth + the python
venv are mounted, so tests/key.json and the dataset tree are unreachable), then grade output/predictions.json with
the task's own tests/grade.py and hidden tests/key.json. Every session goes through the pilot SpendGuard ledger.

Run:   python3 claude_run.py --execute --full-grid --budget-usd 205.00 --max-jobs 2 --per-job-usd 1.00 \
         --max-turns 40 --timeout-seconds 1500 --tasks decode-01a decode-01b
Report: python3 claude_run.py --report
"""
import argparse, json, os, random, shutil, subprocess, sys, tempfile, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, '/home/federicodeponte/sbs-20261023')
from spend_guard import SpendGuard, GuardError, add_arguments  # noqa: E402

H = Path('/home/federicodeponte')
HERE = H / 'work/vals-hyp/broken'
DS = H / 'work/vals-mouse/valkyrie/dataset-broken'
PILOT = H / 'work/vals-hyp/pilot'
OUT = HERE / 'claude'
VENV = H / 'vals-broken-env/py'
TMPROOT = Path('/tmp/claude-broken')
CLAUDE_BIN = '/usr/local/bin/claude'  # real binary; skips the sbin wrapper's forced --chrome
HIDE_IN_DOT_CLAUDE = ['projects', 'backups', 'sessions', 'session-env', 'shell-snapshots', 'debug', 'state', 'cache']
TASKS = [json.loads(l) for l in open(HERE / 'tasks.jsonl')]
META = {t['task_id']: t for t in TASKS}


def job_key(tid, model):
    return f'judge:broken-v3:{tid}:{model}:claude-code'


def bwrap_prefix(wd):
    c = H / '.claude'
    cmd = ['bwrap', '--dev-bind', '/', '/', '--tmpfs', '/home', '--tmpfs', '/tmp', '--tmpfs', '/var/tmp',
           '--bind', str(c), str(c)]
    for d in HIDE_IN_DOT_CLAUDE:
        if (c / d).is_dir():
            cmd += ['--tmpfs', str(c / d)]
    if (c / 'history.jsonl').exists():
        cmd += ['--ro-bind', '/dev/null', str(c / 'history.jsonl')]
    cmd += ['--bind', str(H / '.claude.json'), str(H / '.claude.json'),
            '--ro-bind', str(VENV), str(VENV),
            '--bind', str(wd), str(wd), '--chdir', str(wd),
            '--setenv', 'PATH', f'{VENV}/bin:/usr/local/bin:/usr/bin:/bin',
            '--setenv', 'HOME', str(H), '--setenv', 'TMPDIR', '/tmp', '--']
    return cmd


def grade(tid, pred_file, vdir):
    r = subprocess.run([sys.executable, str(DS / tid / 'tests/grade.py'), str(pred_file), str(DS / tid / 'tests/key.json'),
                        str(vdir)], capture_output=True, text=True, timeout=120)
    rw = (vdir / 'reward.txt').read_text().strip() if (vdir / 'reward.txt').exists() else '0'
    return r, float(rw)


def run(a):
    g = SpendGuard(PILOT, a)
    OUT.mkdir(parents=True, exist_ok=True)
    TMPROOT.mkdir(parents=True, exist_ok=True)
    todo = a.tasks or [t['task_id'] for t in TASKS]
    for tid in todo:
        res_file = OUT / f'{tid}.json'
        key = job_key(tid, a.model)
        if res_file.exists():
            print(tid, 'done, skip', flush=True); continue
        if g.attempted(key):
            print(tid, 'already attempted in ledger without result; not retrying (guard rule)', flush=True); continue
        wd = Path(tempfile.mkdtemp(prefix=f'{tid}-', dir=TMPROOT))
        shutil.copytree(DS / tid / 'environment/data', wd / 'data')
        (wd / 'output').mkdir()
        instr = (DS / tid / 'instruction.md').read_text()
        instr = instr.replace('/app/data/', f'{wd}/data/').replace('/app/output/', f'{wd}/output/')
        assert '/app' not in instr, 'unmapped /app path'
        tdir = OUT / tid; tdir.mkdir(exist_ok=True)
        (tdir / 'prompt.md').write_text(instr)
        try:
            with g.claim(key) as entry:
                cmd = bwrap_prefix(wd) + [CLAUDE_BIN, '-p', '--model', a.model, '--no-session-persistence',
                       '--dangerously-skip-permissions', '--setting-sources', 'project', '--strict-mcp-config',
                       '--disable-slash-commands', '--tools', 'Bash,Read,Write,Edit,Glob,Grep'] + g.claude_flags()
                t0 = time.time()
                r = g.run_process(cmd, input=instr, cwd=wd)
                dt = time.time() - t0
                try:
                    j = json.loads(r.stdout)
                except Exception:  # noqa: BLE001
                    j = {'result': None, 'raw_stdout_tail': r.stdout[-1500:], 'stderr_tail': r.stderr[-1500:]}
                j['returncode'] = r.returncode
                j['wall_s'] = round(dt, 1)
                (tdir / 'claude_session.json').write_text(json.dumps(j, indent=1))
                entry['provider'] = 'claude'
                entry['reported_cost_usd'] = j.get('total_cost_usd')
                # grade exactly like the verifier: missing file -> grader writes 0
                pred = wd / 'output/predictions.json'
                if pred.exists():
                    shutil.copy(pred, tdir / 'predictions.json')
                for extra in wd.iterdir():  # keep the agent's scripts for audit (small)
                    if extra.is_file() and extra.stat().st_size < 200_000:
                        (tdir / 'workdir').mkdir(exist_ok=True); shutil.copy(extra, tdir / 'workdir' / extra.name)
                vdir = tdir / 'verifier'
                gr, reward = grade(tid, pred, vdir)
                rec = {'task_id': tid, 'rewards': {'reward': reward}, 'verifier_result': {'rewards': {'reward': reward}},
                       'metadata': {'reward_source': str(vdir / 'reward.txt'), 'verifier_exit_code': gr.returncode,
                                    'verifier_output': gr.stdout},
                       'agent_caused_exit_reason': None if r.returncode == 0 else f'claude exit {r.returncode}',
                       'task_breakdown': {'agent_run_duration': round(dt, 1)},
                       'claude': {k: j.get(k) for k in ('total_cost_usd', 'num_turns', 'is_error', 'subtype', 'duration_ms',
                                                        'usage', 'returncode', 'stop_reason')},
                       'attempts': 1}
                res_file.write_text(json.dumps(rec, indent=1))
                print(tid, f'reward={reward:.4f}', f'cost={j.get("total_cost_usd")}', f'turns={j.get("num_turns")}',
                      f'rc={r.returncode}', f'{dt:.0f}s', gr.stdout.strip()[:300], flush=True)
        except GuardError as e:
            print('GUARD STOP:', e, flush=True); sys.exit(2)


def report():
    def rows_from(eval_results):
        out = []
        for tid, x in eval_results.items():
            v = json.loads(x['metadata']['verifier_output'].strip().splitlines()[-1])
            out.append(dict(tid=tid, twin=META[tid]['twin'], fault=META[tid]['fault'], acc=v['acc'],
                            claimed=v.get('claimed_accuracy'), gap=v.get('gap'), named=v.get('fault_named'),
                            fa=v.get('false_alarm'), reward=x['rewards']['reward']))
        return out
    codex = []
    for f in sorted((HERE / 'full').glob('*.json')):
        codex += rows_from(json.load(open(f))['evaluation_results'])
    claude = rows_from({p.stem: json.load(open(p)) for p in sorted(OUT.glob('decode-*.json'))})

    def agg(rows):
        cells = defaultdict(list)
        for r in rows:
            cells[(r['twin'], r['fault'])].append(r)
        res = {}
        for k, rs in sorted(cells.items()):
            m = lambda f: sum((r[f] or 0) for r in rs) / len(rs)
            res[k] = dict(n=len(rs), claimed=m('claimed'), acc=m('acc'), gap=m('gap'),
                          named=sum(1 for r in rs if r['named']), fa=sum(1 for r in rs if r['fa']), reward=m('reward'))
        for tw in ('clean', 'defect'):
            rs = [r for r in rows if r['twin'] == tw]
            if rs:
                res[(tw, 'ALL')] = dict(n=len(rs), claimed=sum(r['claimed'] or 0 for r in rs) / len(rs),
                                        acc=sum(r['acc'] for r in rs) / len(rs), named=sum(1 for r in rs if r['named']),
                                        fa=sum(1 for r in rs if r['fa']), reward=sum(r['reward'] for r in rs) / len(rs))
                res[(tw, 'ALL')]['gap'] = sum(r['gap'] or 0 for r in rs) / len(rs)
        res[('all', 'ALL')] = dict(n=len(rows), reward=sum(r['reward'] for r in rows) / max(1, len(rows)))
        return res
    A, B = agg(claude), agg(codex)
    print('| twin | fault | Claude n | claimed | actual | gap | named | false alarms | reward | Codex n | claimed | actual | gap | named | false alarms | reward |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    f2 = lambda d, k: ('' if d is None or k not in d else (f'{d[k]:.2f}' if isinstance(d[k], float) else str(d[k])))
    for k in sorted(set(A) | set(B), key=lambda k: (k[0] == 'all', k[0], k[1] == 'ALL', k[1])):
        a, b = A.get(k), B.get(k)
        cols = lambda d: [f2(d, x) for x in ('n', 'claimed', 'acc', 'gap', 'named', 'fa', 'reward')]
        print('| ' + ' | '.join([k[0], k[1]] + cols(a) + cols(b)) + ' |')
    tot = sum(json.load(open(p))['claude'].get('total_cost_usd') or 0 for p in OUT.glob('decode-*.json'))
    print(f'\nClaude tasks graded: {len(claude)}; total reported cost ${tot:.2f}; Codex rows: {len(codex)} (3 passes)')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); add_arguments(p)
    p.add_argument('--tasks', nargs='*'); p.add_argument('--model', default='opus'); p.add_argument('--report', action='store_true')
    a = p.parse_args()
    report() if a.report else run(a)
