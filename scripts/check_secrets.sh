#!/usr/bin/env bash
# Fail if anything that looks like a credential is tracked or present in history.
set -euo pipefail
cd "$(dirname "$0")/.."
PATTERNS='sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|(TYPESAFE|OPENAI|ANTHROPIC)_API_KEY\s*=\s*[A-Za-z0-9]|Bearer [A-Za-z0-9_.-]{20,}'
fail=0
# `grep -q` prints nothing, so piping it into another grep tested an empty stream and the
# branch could never fire. Filter first, then test what is left.
if git ls-files | grep -E '(^|/)\.env($|\.)' | grep -qv '\.env\.example'; then echo "tracked .env file"; fail=1; fi
if git ls-files -z | xargs -0 grep -nE "$PATTERNS" 2>/dev/null | grep -v '.env.example' | grep -v 'check_secrets.sh'; then echo "credential-like string in tracked files"; fail=1; fi
if git log -p --all | grep -E "$PATTERNS" | grep -v 'check_secrets.sh' | head -1 | grep -q .; then echo "credential-like string in git history"; fail=1; fi
[ $fail -eq 0 ] && echo "secrets: clean" || exit 1
