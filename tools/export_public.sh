#!/usr/bin/env bash
set -euo pipefail

SRC="${1:?usage: export_public.sh <path-to-private-forex-repo>}"
DEST="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$DEST/src"

EXCLUDES=(
  --exclude 'target/'            # Rust build output (14 GB)
  --exclude '__pycache__/'  --exclude '*.pyc'  --exclude '*.pyo'
  --exclude '.env'               # credentials (.env.example is kept
  --exclude '.idea/'  --exclude '.vscode/'  --exclude '.DS_Store'
  --exclude '.pytest_cache/'  --exclude '.hypothesis/'  --exclude '.ruff_cache/'
  --exclude 'cache/'             # downloaded market-data caches
  --exclude '*.parquet'          # market-data files (fetched, not source)
  --exclude '*.mov'  --exclude '*.mp4'        # screen recordings
  --exclude '*.log'
  --exclude '/service/'          # ta/service: uncommitted scaffolding, not part of the report
  --exclude 'specs/'             # this repository is code only) no specs or documents
)

rm -rf "$OUT"; mkdir -p "$OUT"

copy() { # copy <relative-src> <dest-name>
  echo "  $1 -> src/$2"
  rsync -a "${EXCLUDES[@]}" "$SRC/$1/" "$OUT/$2/"
}

echo "exporting source ..."
copy modelenv                    modelenv
copy ta                          ta
copy deepqnetwork                deepqnetwork
copy probabilisticforecaster     probabilisticforecaster
copy tradingmodel/intraday/dqnpf dqnpf
copy marketdata                  marketdata
copy commons                     commons

echo "exporting the feature prototype ..."
rsync -a "${EXCLUDES[@]}" \
  "$SRC/finalreport/preliminaryreport/prototype/" "$OUT/feature-prototype/"

echo
echo "sizes:"; du -sh "$OUT"/* | sed 's/^/  /'
echo
"$DEST/tools/scan_secrets.sh" "$DEST"
