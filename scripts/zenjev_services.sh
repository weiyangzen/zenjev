#!/usr/bin/env bash
# ZenJev perpetual service control (§1.6, ZJ-072).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
UNITS=(zenjev-feeder zenjev-loop zenjev-fabricator zenjev-serve zenjev-deploy zenjev-console)
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
PYTHON="$REPO/.venv/bin/python"
LOG_DIR="$REPO/runs/jev/logs"
MAX_LOG_MB="${MAX_LOG_MB:-20}"

command="${1:-status}"
shift || true

case "$command" in
  install)
    exec bash "$REPO/scripts/install_zenjev_services.sh"
    ;;
  start)
    systemctl --user start "${UNITS[@]}"
    ;;
  stop)
    systemctl --user stop "${UNITS[@]}"
    ;;
  restart)
    systemctl --user restart "${UNITS[@]}"
    ;;
  enable)
    systemctl --user enable "${UNITS[@]}"
    ;;
  disable)
    systemctl --user disable "${UNITS[@]}"
    ;;
  status)
    systemctl --user --no-pager --lines=0 status "${UNITS[@]}" || true
    echo
    echo "console: http://$(hostname -I | awk '{print $1}'):8790/"
    ;;
  logs)
    unit="${1:-zenjev-loop}"; lines="${2:-60}"
    if [[ ! " ${UNITS[*]} " =~ " ${unit} " ]]; then
      echo "unknown unit: ${unit}; expected one of ${UNITS[*]}" >&2
      exit 2
    fi
    systemctl --user --no-pager -n "$lines" -u "$unit"
    ;;
  rotate)
    for unit in "${UNITS[@]}"; do
      file="$LOG_DIR/$unit.log"
      [[ -s "$file" ]] || continue
      size_mb=$(( $(stat -c%s "$file") / 1048576 ))
      if (( size_mb > MAX_LOG_MB )); then
        mv -f "$file" "$file.1"
        echo "rotated $file (${size_mb}MB -> $file.1)"
      fi
      : > "$file"
    done
    echo "rotated service logs (journald keeps its own history)"
    ;;
  health)
    cd "$REPO" && exec "$PYTHON" -m jev.cli health "$@"
    ;;
  evidence)
    cd "$REPO" && exec "$PYTHON" scripts/perpetual_evidence.py
    ;;
  deploy)
    cd "$REPO" && exec "$PYTHON" scripts/zenjev_deploy.py --once "$@"
    ;;
  *)
    cat <<EOF
usage: zenjev_services.sh <command>

  install    install/refresh the systemd user units and restart them
  start      start all perpetual services
  stop       stop all perpetual services
  restart    restart all perpetual services (resume from checkpoint + watermark)
  enable     enable at login/linger
  disable    disable at login/linger
  status     unit status plus the console URL
  logs       logs [unit] [lines]        default: zenjev-loop 60
  rotate     rotate service log files over MAX_LOG_MB (default 20)
  health     heartbeat/gap/session report (exit != 0 when a service is stale)
  evidence   freeze G12-G15 evidence into artifacts/perpetual/acceptance.json
  deploy     evaluate one LoRA deploy decision (add --force for a drill)
EOF
    exit 2
    ;;
esac
