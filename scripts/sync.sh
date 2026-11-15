#!/usr/bin/env bash
# Sync working dirs on AX41 into the brain-eval repo and push. Aborts on anything that looks like a secret.
set -euo pipefail
R=$HOME/work/brain-eval-repo
X=(--exclude=.venv --exclude=venv --exclude=_uvcache --exclude=.uvcache --exclude=_data --exclude=_tools --exclude=_raw
   --exclude=data/ --exclude=bin/ --exclude='*.nwb' --exclude=__pycache__ --exclude='.claude-*' --exclude='.spend-*'
   --exclude=empty/ --exclude=.cache --exclude=node_modules --exclude='*.h5' --exclude='*.npz' --exclude='*.npy'
   --exclude='*.parquet' --exclude='*.edf' --exclude=.git --exclude=smell/ --exclude=.docker --exclude=.xdg --exclude=.local
   --exclude=.env --exclude='*.log' --exclude=.DS_Store --exclude=Valkyrie/ --exclude=public-agent-registry/ --max-size=15M)
rsync -a "${X[@]}" "$HOME/work/vals-mouse/" "$R/mouse/"
rsync -a "${X[@]}" "$HOME/work/vals-hyp/" "$R/hypotheses/"
cd "$R"
if grep -rIlE "(sk-ant-[A-Za-z0-9_-]{20}|sk-proj-[A-Za-z0-9_-]{20}|xox[bpcs]-[0-9]|ghp_[A-Za-z0-9]{30}|gho_[A-Za-z0-9]{30}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY)" --exclude-dir=.git --exclude=DMS_substitutions.csv . ; then
  echo "SECRET-LIKE STRING FOUND, not pushing"; exit 1; fi
git add -A
git diff --cached --quiet && { echo "nothing new"; exit 0; }
git -c user.name="Federico De Ponte" -c user.email="depontefede@gmail.com" commit -qm "Sync working files $(date -u +%Y-%m-%dT%H:%MZ)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -q && echo "pushed $(git rev-parse --short HEAD)"
