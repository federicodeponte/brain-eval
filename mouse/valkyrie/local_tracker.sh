#!/bin/bash
# Local Valkyrie Tracker on AX41 (ports remapped: 8000/8002/5432 are taken on this host).
# Tracker http://127.0.0.1:8031, control 8035, postgres 55432, projects 8010. No Vals login needed.
set -e
export DOCKER_CONFIG=$HOME/work/vals-mouse/.docker
export PATH=$HOME/work/vals-mouse/bin:$PATH UV_CACHE_DIR=$HOME/work/vals-mouse/.uvcache
HERE=$HOME/work/vals-mouse/valkyrie
cd $HERE/Valkyrie/services/tracker
export EXECUTOR_BOOTSTRAP_KEY=${EXECUTOR_BOOTSTRAP_KEY:-local-bootstrap-$(hostname)}
export PROJECT_SERVICE_TOKEN=local-project-token
DB=postgresql://tracker:tracker@localhost:55432/tracker
docker compose -f docker-compose.yml -f $HERE/compose.override.yml up -d --wait postgres projects
DATABASE_URL=$DB uv run alembic upgrade head
export EXECUTOR_PAYLOAD_LOCAL_KEY=$(uv run python -c "import base64,os;print(base64.b64encode(os.urandom(32)).decode())")
export EXECUTOR_LAUNCHER=local PROJECT_SERVICE_URL=http://127.0.0.1:8010
unset EXECUTOR_PAYLOAD_KMS_KEY_ID
DATABASE_URL=$DB uv run uvicorn tracker.executor.control_app:app --host 127.0.0.1 --port 8035 &
cpid=$!; trap "kill $cpid" EXIT
DATABASE_URL=$DB EXECUTOR_CONTROL_URL=http://127.0.0.1:8035 uv run python -m tracker.serve --config $HERE/local-server.yaml --port 8031
