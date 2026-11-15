# Claude Opus (Claude Code, 1 pass) on gym v4, 45 environments, fixed grader gym/grade.py

| group | n | claimed | real | reward | faults named | false alarms |
|---|---|---|---|---|---|---|
| clean | 15 | 68% | 72% | 0.85 | - | 0/15 |
| single fault | 15 | 63% | 59% | 0.73 | 13/15 | - |
| combined | 15 | 71% | 61% | 0.68 | 21/30 | - |
| **overall** | 45 | | | **0.756** | | |

Named per fault: dead 9/9, leak 2/10, noise 8/8, perm 6/9, shift 9/9

Codex (gpt-5.3-codex, 3 passes, same grader): overall 0.452; combined claimed 79% real 24%. See gym/full_results.md.
