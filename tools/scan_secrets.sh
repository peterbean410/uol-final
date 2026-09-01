#!/usr/bin/env bash
set -uo pipefail
ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
echo "secret scan over $ROOT"

FAIL=0
report() { echo "  !! $1"; FAIL=1; }

while IFS= read -r f; do report "sensitive file: ${f#$ROOT/}"; done < <(
  find "$ROOT" -type f \( -name '.env' -o -name 'kubeconfig' -o -name '*.pem' \
       -o -name '*.p12' -o -name 'id_rsa*' -o -name '*accessKeys*.csv' \) \
       -not -path '*/.git/*' 2>/dev/null)

PATTERNS=(
  'AKIA[0-9A-Z]{16}'                                  # AWS access key id
  '(aws_secret_access_key|password|passwd|api_key|secret_key|access_token)[[:space:]]*=[[:space:]]*.?["'"'"'][A-Za-z0-9/+=_.-]{20,}["'"'"']'  # literal credential
  '-----BEGIN [A-Z ]*PRIVATE KEY-----'
  'https://[A-Za-z0-9._%-]+:[A-Za-z0-9._%-]{16,}@'    # creds embedded in a URL
  'xox[baprs]-[0-9A-Za-z-]{10,}'                      # slack
  'sk-[A-Za-z0-9]{32,}'                               # openai-style key
  'ghp_[0-9A-Za-z]{36}'                               # github PAT
)
for p in "${PATTERNS[@]}"; do
  while IFS= read -r hit; do report "pattern /$p/ in ${hit#$ROOT/}"; done < <(
    grep -rIlE "$p" "$ROOT" --exclude-dir=.git --exclude-dir=tools 2>/dev/null)
done

if [ "$FAIL" -eq 0 ]; then echo "  clean (nothing found"; else
  echo; echo "SECRET SCAN FAILED) do not push."; exit 1; fi
