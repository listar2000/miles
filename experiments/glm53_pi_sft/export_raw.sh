#!/usr/bin/env bash
# Download every trial (all outcomes) of each finished GLM-5.3-Flash Pi eval, once per eval id.
# Usage: export_raw.sh <eval-uuid>...   -> ~/data/sft-glm53-pi/raw/<uuid>/ (Harbor layout + manifest.json) and .done
# Mobius reports complete=false when some trials have no trace (status "missing", runtime errors); that is
# accepted as long as every trace the manifest lists as available is on disk.
set -u
DATA=$HOME/data/sft-glm53-pi  # the working copy on the devbox
mkdir -p "$DATA/raw" && cd "$DATA"
check() {  # every available trace in the manifest exists
  python3 - "$1" <<'P'
import json, os, sys
d = sys.argv[1]; m = json.load(open(f"{d}/manifest.json"))
bad = [t for t in m["trials"] if t["trace"]["status"] == "available"
       and not all(os.path.exists(f'{d}/{t["trace"]["folder"]}/{f["path"]}') for f in t["trace"]["files"])]
print(f'{len(m["trials"])} trials, {sum(t["trace"]["status"] != "available" for t in m["trials"])} without trace, {len(bad)} missing on disk')
sys.exit(1 if bad else 0)
P
}
for id in "$@"; do
  out=raw/$id
  [ -f "$out/.done" ] && { echo "skip $id"; continue; }
  rm -rf "$out.part"  # an interrupted or incomplete download starts over
  mobius eval traces "$id" --download-dir "$out.part" > "raw/$id.log" 2>&1
  if [ -f "$out.part/manifest.json" ] && check "$out.part"; then
    mv "$out.part" "$out" && date -u +%FT%TZ > "$out/.done" && echo "done $id $(du -sh "$out" | cut -f1)"
  else
    echo "FAIL $id (see raw/$id.log)"
  fi
done
