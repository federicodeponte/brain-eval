# Status (updated by Claude for Federico + cosmicz)

**Submissions close Sun Nov 15, 2pm PT.** Demo page: `demo/index.html` (open in a browser).

## Done
- Main sets: human neurons (DANDI 000004, 40 Qs), mouse K=4/K=8 (Allen, `mouse/tasks_v2.jsonl` = leak-fixed)
- 5 learning controls built (`hypotheses/controls/`), Valkyrie dataset folders in `mouse/valkyrie/dataset-*`
- Pilot on Claude (n=10/task): human Haiku 5/10, Opus 5/10, decoder 65%; mouse K=4 Haiku 6/10, Opus 8/10, decoder 87%; mouse K=8 Opus 7/10, decoder 67%
- Controls (Opus): label-swap follows examples 12/20 vs prior 3/20; 0-shot = always "A" (chance), 1-shot 7/10
- Raw per-question results: `hypotheses/pilot/results.jsonl`

## Blocked
- Vals hosted runs need our login (depontefede@gmail.com, already logged in on AX41) added to the hackathon Valkyrie org. Asked in Slack #build-your-own-eval-track; Luma registration still pending approval.

## Next
1. Once in the org: push `dataset-human`, `dataset-v2`, controls; run 3 models (commands in `mouse/valkyrie/STATE.md`)
2. Remaining controls on hosted run: swap-out, null, planted, full shot curve (3/5 shots)
3. Optional second act: pulse-from-webcam (`hypotheses/pulse/`)

## Sync
`scripts/sync.sh` on AX41 copies working dirs here and pushes (every 15 min via cron, with a secret scan).
