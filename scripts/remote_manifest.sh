#!/usr/bin/env bash
set -euo pipefail
root="${1:-.}"
find "$root" -type f \
  -not -path '*/.git/*' \
  -not -path '*/.zenjev/runtime/*' \
  -print0 | sort -z | xargs -0 sha256sum

