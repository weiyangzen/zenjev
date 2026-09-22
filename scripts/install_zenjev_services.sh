#!/usr/bin/env bash
# Install the perpetual ZenJev services as user systemd units (§1.6).
# Idempotent: rewrites the unit files, reloads, enables and restarts.
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
PYTHON="$REPO/.venv/bin/python"
LOG_DIR="$REPO/runs/jev/logs"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

mkdir -p "$UNIT_DIR" "$LOG_DIR"

write_unit() {
  local name="$1" description="$2" exec_line="$3"
  cat > "$UNIT_DIR/$name.service" <<EOF
[Unit]
Description=$description
After=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
WorkingDirectory=$REPO
Environment=JEV_MODEL_PATH=$HOME/jev-model-base
Environment=PYTHONUNBUFFERED=1
Environment=JEV_MQ_BRIDGE_BIN=$REPO/mq/target/release/jev-mq-bridge
ExecStart=$exec_line
Restart=always
RestartSec=5
KillMode=mixed
TimeoutStopSec=30
StandardOutput=append:$LOG_DIR/$name.log
StandardError=append:$LOG_DIR/$name.log

[Install]
WantedBy=default.target
EOF
}

write_unit "zenjev-feeder" "ZenJev raw dump feeder (perpetual dir-spool producer)" \
  "$PYTHON $REPO/scripts/zenjev_feeder.py --max-records 40 --tick-sleep 0.5"
write_unit "zenjev-loop" "ZenJev perpetual MQ + RSI LoRA training + EMA inference loop" \
  "$PYTHON $REPO/scripts/zenjev_loop.py"
write_unit "zenjev-fabricator" "ZenJev synthetic schema-task fabricator (canaries)" \
  "$PYTHON $REPO/scripts/zenjev_fabricator.py --per-tick 2 --interval 2"
write_unit "zenjev-serve" "ZenJev perpetual inference service (deployed LoRA, 0.0.0.0:8791)" \
  "$PYTHON $REPO/scripts/zenjev_serve.py"
write_unit "zenjev-deploy" "ZenJev LoRA deploy gate (redeploy zenjev-serve on newer stable generations)" \
  "$PYTHON $REPO/scripts/zenjev_deploy.py"
write_unit "zenjev-console" "ZenJev live console (read-only web panel)" \
  "$PYTHON $REPO/scripts/zenjev_console.py"

systemctl --user daemon-reload
systemctl --user enable zenjev-feeder zenjev-loop zenjev-fabricator zenjev-serve zenjev-deploy zenjev-console

if ! loginctl show-user "$USER" 2>/dev/null | grep -q "Linger=yes"; then
  echo "note: loginctl enable-linger $USER is required for reboot persistence (needs sudo)"
fi

systemctl --user restart zenjev-feeder zenjev-loop zenjev-fabricator zenjev-serve zenjev-deploy zenjev-console
sleep 2
systemctl --user --no-pager --lines=0 status zenjev-feeder zenjev-loop zenjev-fabricator zenjev-serve zenjev-deploy zenjev-console || true
echo "installed: $UNIT_DIR/zenjev-{feeder,loop,fabricator,serve,deploy,console}.service"
