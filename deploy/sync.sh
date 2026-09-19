#!/usr/bin/env bash
# Copy the invalidate package and the playground page into deploy/ so Vercel can upload them.
# Re-run after any change under src/invalidate. Never symlink: Vercel uploads do not follow links reliably.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/../src/invalidate"

rsync -a --delete \
  --exclude '__pycache__' --exclude '*.pyc' \
  --exclude 'ui/static/' \
  "$SRC/" "$HERE/invalidate/"

mkdir -p "$HERE/public"
cp "$SRC/ui/static/playground.html" "$HERE/public/index.html"

echo "synced -> $HERE/invalidate and $HERE/public/index.html"
