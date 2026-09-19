#!/usr/bin/env bash
# Copy the invalidate package and the two pages into deploy/ so Vercel can upload them.
# Re-run after any change under src/invalidate. Never symlink: Vercel uploads do not follow links reliably.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/../src/invalidate"

rsync -a --delete \
  --exclude '__pycache__' --exclude '*.pyc' \
  --exclude 'ui/static/' \
  "$SRC/" "$HERE/invalidate/"

mkdir -p "$HERE/public"
cp "$SRC/ui/static/chat.html" "$HERE/public/index.html"        # landing page: the conversation demo
cp "$SRC/ui/static/playground.html" "$HERE/public/paste.html"  # two-box paste mode, served at /paste

echo "synced -> $HERE/invalidate, $HERE/public/index.html (chat), $HERE/public/paste.html (playground)"
