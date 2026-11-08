#!/bin/bash
# Watch twin jobs (one per cluster) and cancel the loser. State file lines: "<task> <helios_id> <athena_id>".
# The first cluster whose job is RUNNING wins (its twin is scancel'd); if both are RUNNING at one poll the
# later-started one loses. A job that vanishes from the queue without ever running (cancelled/failed) does
# NOT cancel its twin. Exits when every pair is settled. Run from the laptop; polls every POLL seconds.
set -uo pipefail
STATE="${1:?usage: flops_race.sh <state file>}"; POLL="${POLL:-30}"
declare -A settled
q() {  # q <cluster> <jobid> -> "<state> <start-time>" from squeue, empty if the job left the queue,
       # "ERR" if ssh itself failed (exit 255) — must NOT be read as "left the queue"
  local out rc
  out=$(ssh -o ConnectTimeout=20 -o BatchMode=yes "$1" "squeue -h -j $2 -o '%t %S' 2>/dev/null" 2>/dev/null); rc=$?
  if [[ $rc -eq 255 ]]; then echo ERR; else echo "$out" | head -1; fi
}
final() {  # final <cluster> <jobid> -> sacct state of a job that left the queue
  ssh -o ConnectTimeout=20 -o BatchMode=yes "$1" "sacct -n -X -j $2 -o State%20 2>/dev/null" 2>/dev/null | head -1 | tr -d ' '
}
while true; do
  remaining=0
  while read -r task hid aid; do
    [[ -z "${task:-}" || -n "${settled[$task]:-}" ]] && continue
    hq=$(q helios "$hid"); aq=$(q athena "$aid"); hs=${hq%% *}; as=${aq%% *}
    echo "$(date +%H:%M:%S) ${task}: helios[$hid]=${hs:-gone} athena[$aid]=${as:-gone}"
    if [[ "$hs" == "ERR" || "$as" == "ERR" ]]; then
      echo "  ssh failed -- state unknown, retrying next poll"; remaining=$((remaining+1)); continue
    fi
    if [[ "$hs" == "R" && "$as" == "R" ]]; then
      if [[ "${hq#* }" > "${aq#* }" ]]; then ssh helios "scancel $hid"; echo "  both running: cancelled helios $hid (started later)"
      else ssh athena "scancel $aid"; echo "  both running: cancelled athena $aid (started later)"; fi
      settled[$task]=1
    elif [[ "$hs" == "R" ]]; then ssh athena "scancel $aid"; echo "  helios started first -> cancelled athena $aid"; settled[$task]=1
    elif [[ "$as" == "R" ]]; then ssh helios "scancel $hid"; echo "  athena started first -> cancelled helios $hid"; settled[$task]=1
    elif [[ -z "$hs" && -n "$as" ]]; then
      hf=$(final helios "$hid")
      if [[ "$hf" == COMPLETED* ]]; then ssh athena "scancel $aid"; echo "  helios already COMPLETED -> cancelled athena $aid"
      else echo "  helios job left the queue (${hf:-?}); athena twin keeps going"; fi
      settled[$task]=1
    elif [[ -z "$as" && -n "$hs" ]]; then
      af=$(final athena "$aid")
      if [[ "$af" == COMPLETED* ]]; then ssh helios "scancel $hid"; echo "  athena already COMPLETED -> cancelled helios $hid"
      else echo "  athena job left the queue (${af:-?}); helios twin keeps going"; fi
      settled[$task]=1
    elif [[ -z "$hs" && -z "$as" ]]; then echo "  neither job is queued any more"; settled[$task]=1
    else remaining=$((remaining+1)); fi
  done < "$STATE"
  [[ $remaining -eq 0 ]] && { echo "all pairs settled"; break; }
  sleep "$POLL"
done
