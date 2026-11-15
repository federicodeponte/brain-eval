import json, time
from pathlib import Path
b = Path.home()/'work/vals-hyp/pilot'; b.mkdir(parents=True, exist_ok=True)
json.dump({"approved_by": "Federico",
 "approval_reference": "2026-11-14 chat: Federico chose 'Both' to the question proposing a subscription pilot of ~100 claude -p calls (10 questions per hypothesis x 1 model, 1 worker) on AX41, then said 'dont wait just go, be proactive'.",
 "max_total_jobs": 100, "estimated_allowance_usd": "15.00", "per_job_usd": "0.50",
 "job_prefix": "judge:", "expires_at": time.time() + 20*3600}, open(b/'.claude-full-grid-approval.json','w'), indent=2)
print('ok')
