#!/bin/bash
# Cleanup of leftover ray cluster / verl trainer / vLLM workers from a previous
# phase, so the next `verl.trainer.main_ppo` starts a *fresh* ray cluster
# without colliding with stale TCPStore endpoints, dashboard ports, etc. Ray's
# head daemons (gcs_server, raylet, dashboard_agent, ...) survive a `pkill
# 'ray::'` and hold the ports the new cluster's allocator wants.
#
# SCOPING: kills are restricted to PIDs that (a) hold a CUDA context on
# RZERO_GPUS (default 0,1,2,3) per nvidia-smi, or (b) are descendants /
# pattern-matching ancestors of those. PIDs on other GPUs are *never* killed —
# a bare `pkill -f '<pattern>'` would hit other users' / other runs' processes.
#
# Usage: bash cleanup_between_phases.sh [tag]
#   tag: free-form string for log lines (e.g. "post-challenger", "solver-entry").
set -u

TAG=${1:-"cleanup"}
echo "[$TAG] starting scoped cleanup"

# --- 0. Define the GPU scope we are allowed to touch ----------------------
# GPU indices belonging to *our* run: a fixed 4-GPU layout on 0-3
# (trainer 0,1 + answers server 2 + RM 3).
RZERO_GPUS="${RZERO_GPUS:-0,1,2,3}"
echo "[$TAG] scope: GPUs $RZERO_GPUS"

# Seed set: PIDs holding a CUDA context on RZERO_GPUS, per nvidia-smi. Expanded
# with descendants/ancestors further down.
collect_seed_pids() {
    local seed=()
    local mine="$$"  # never list our own shell as a target
    declare -A UUID2IDX
    while IFS=',' read -r idx uuid; do
        idx="${idx// /}"; uuid="${uuid// /}"
        [[ -z "$idx" ]] && continue
        UUID2IDX["$uuid"]="$idx"
    done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null)

    seed=()
    while IFS=',' read -r pid uuid; do
        pid="${pid// /}"; uuid="${uuid// /}"
        [[ -z "$pid" ]] && continue
        [[ "$pid" == "$mine" ]] && continue
        local idx="${UUID2IDX[$uuid]:-}"
        [[ -z "$idx" ]] && continue
        if [[ ",$RZERO_GPUS," == *",$idx,"* ]]; then
            seed+=("$pid")
        fi
    done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null)

    printf '%s\n' "${seed[@]}" | sort -u
}

# Expand a PID set with all descendants (iterated to fixed point). Downward
# only — we never walk up towards init/systemd here.
expand_descendants() {
    local input
    input=$(cat)
    local frontier="$input"
    local seen="$input"
    while [[ -n "$frontier" ]]; do
        local next=""
        for pid in $frontier; do
            local kids
            kids=$(pgrep -P "$pid" 2>/dev/null || true)
            for k in $kids; do
                if ! grep -qx "$k" <<< "$seen"; then
                    seen=$(printf '%s\n%s' "$seen" "$k")
                    next=$(printf '%s\n%s' "$next" "$k")
                fi
            done
        done
        frontier=$(echo -n "$next" | sed '/^$/d')
    done
    echo "$seen" | sed '/^$/d' | sort -u
}

# Process patterns we recognise as ours. Only used to *gate* which ancestors
# get pulled in, so unrelated daemons are never swept up.
PATTERNS_RX='ray::|VLLM::EngineCore|start_vllm_server_answers\.py|start_rm_server\.py|verl\.trainer\.main_ppo|verl\.model_merger|gcs_server|raylet|dashboard_agent\.py|log_monitor\.py|monitor\.py.*--logs-dir|default_worker\.py|multiprocessing\.resource_tracker'

# Ray/verl ancestors of GPU-holding workers are CPU-only and never appear in
# nvidia-smi (e.g. the verl.trainer.main_ppo parent of a VLLM::EngineCore).
# One hop up, gated by PATTERNS_RX.
add_matching_ancestors() {
    local input
    input=$(cat)
    local out="$input"
    for pid in $input; do
        local parent
        parent=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
        [[ -z "$parent" || "$parent" == "1" || "$parent" == "0" ]] && continue
        local pcmd
        pcmd=$(ps -o args= -p "$parent" 2>/dev/null)
        if [[ -n "$pcmd" ]] && echo "$pcmd" | grep -Eq "$PATTERNS_RX"; then
            if ! grep -qx "$parent" <<< "$out"; then
                out=$(printf '%s\n%s' "$out" "$parent")
            fi
        fi
    done
    echo "$out" | sed '/^$/d' | sort -u
}

# Compose the final kill set.
SEED=$(collect_seed_pids)
if [[ -z "$SEED" ]]; then
    echo "[$TAG] no GPU-holding PIDs on RZERO_GPUS=$RZERO_GPUS"
fi

# Pattern-matching PIDs, so a verl.trainer.main_ppo whose vLLM child was
# already killed (hence no GPU context) is still included.
PATTERN_PIDS=$(pgrep -f "$PATTERNS_RX" 2>/dev/null | sort -u || true)

# Keep only pattern PIDs whose ancestor chain reaches a GPU-holding seed —
# without that check we would match unrelated vLLM engines on other GPUs.
keep_in_scope() {
    local input
    input=$(cat)
    local kept=""
    local seeds_set
    seeds_set=$(echo "$SEED")
    for pid in $input; do
        local cur="$pid"
        local accepted=0
        for _ in 1 2 3 4 5 6; do
            [[ -z "$cur" || "$cur" == "1" || "$cur" == "0" ]] && break
            if grep -qx "$cur" <<< "$seeds_set"; then
                accepted=1; break
            fi
            cur=$(ps -o ppid= -p "$cur" 2>/dev/null | tr -d ' ')
        done
        if [[ $accepted -eq 1 ]]; then
            kept=$(printf '%s\n%s' "$kept" "$pid")
        fi
    done
    echo "$kept" | sed '/^$/d' | sort -u
}

PATTERN_IN_SCOPE=""
if [[ -n "$PATTERN_PIDS" && -n "$SEED" ]]; then
    PATTERN_IN_SCOPE=$(echo "$PATTERN_PIDS" | keep_in_scope)
fi

# Union seed + in-scope pattern PIDs, then expand descendants and ancestors.
INITIAL=$(printf '%s\n%s' "$SEED" "$PATTERN_IN_SCOPE" | sed '/^$/d' | sort -u)
if [[ -z "$INITIAL" ]]; then
    KILL_SET=""
else
    KILL_SET=$(echo "$INITIAL" | expand_descendants | add_matching_ancestors)
fi

# Exclude candidates whose ancestor chain matches RZERO_KILL_EXCLUDE_RX. Set it
# when co-running another experiment on this node (e.g. 'g_zero'): nvidia-smi
# can transiently attribute a foreign pid to our GPU set. Matching on ancestors
# because the vLLM engine procs carry no project name in argv.
if [[ -n "${RZERO_KILL_EXCLUDE_RX:-}" && -n "$KILL_SET" ]]; then
    GUARDED=""
    for pid in $KILL_SET; do
        gcur="$pid"; gskip=0
        for _ in 1 2 3 4 5 6; do
            [[ -z "$gcur" || "$gcur" == "1" || "$gcur" == "0" ]] && break
            if ps -o args= -p "$gcur" 2>/dev/null | grep -Eq "$RZERO_KILL_EXCLUDE_RX"; then
                gskip=1; break
            fi
            gcur=$(ps -o ppid= -p "$gcur" 2>/dev/null | tr -d ' ')
        done
        if [[ $gskip -eq 1 ]]; then
            echo "[$TAG] EXCLUDE: skipping pid=$pid (matches RZERO_KILL_EXCLUDE_RX ancestry)"
        else
            GUARDED=$(printf '%s\n%s' "$GUARDED" "$pid")
        fi
    done
    KILL_SET=$(echo "$GUARDED" | sed '/^$/d' | sort -u)
fi

# Never kill our own shell or its ancestors.
SELF_CHAIN="$$"
cur="$$"
for _ in 1 2 3 4 5 6 7 8; do
    cur=$(ps -o ppid= -p "$cur" 2>/dev/null | tr -d ' ')
    [[ -z "$cur" || "$cur" == "1" || "$cur" == "0" ]] && break
    SELF_CHAIN=$(printf '%s\n%s' "$SELF_CHAIN" "$cur")
done

FINAL_KILL=""
for pid in $KILL_SET; do
    if grep -qx "$pid" <<< "$SELF_CHAIN"; then
        echo "[$TAG] skip self-chain pid $pid"
        continue
    fi
    FINAL_KILL=$(printf '%s\n%s' "$FINAL_KILL" "$pid")
done
FINAL_KILL=$(echo "$FINAL_KILL" | sed '/^$/d' | sort -u)

# --- 1. Shut down *our* ray daemons (scoped by RAY_TMPDIR session path) ----
# Never `ray stop --force` here: it matches ray component *process names* for
# the whole user account and can SIGKILL an unrelated run's raylet mid-training.
# Ray head daemons embed their session path (under RAY_TMPDIR) in argv, so
# pkill -f on that path hits exactly this run's cluster.
RAY_PAT=${RAY_TMPDIR:-/tmp/ray_rzero}
# That pattern also matches OUR OWN ancestor shells when a launcher was invoked
# with RAY_TMPDIR=... inline, so enumerate matches and skip self + ancestors.
RAY_PIDS=$(pgrep -f "$RAY_PAT" 2>/dev/null || true)
if [[ -n "$RAY_PIDS" ]]; then
    SELF_ANCESTORS="$$"; acur="$$"
    for _ in 1 2 3 4 5 6 7 8; do
        acur=$(ps -o ppid= -p "$acur" 2>/dev/null | tr -d ' ')
        [[ -z "$acur" || "$acur" == "0" || "$acur" == "1" ]] && break
        SELF_ANCESTORS="$SELF_ANCESTORS $acur"
    done
    KILLED_ANY=0
    for rp in $RAY_PIDS; do
        skip=0
        for a in $SELF_ANCESTORS; do [[ "$rp" == "$a" ]] && { skip=1; break; }; done
        [[ $skip -eq 1 ]] && { echo "[$TAG] skipping ray-pattern pid $rp (self/ancestor)"; continue; }
        kill -TERM "$rp" 2>/dev/null || true; KILLED_ANY=1
    done
    if [[ $KILLED_ANY -eq 1 ]]; then
        sleep 3
        for rp in $RAY_PIDS; do
            skip=0
            for a in $SELF_ANCESTORS; do [[ "$rp" == "$a" ]] && { skip=1; break; }; done
            [[ $skip -eq 0 ]] && kill -KILL "$rp" 2>/dev/null || true
        done
        echo "[$TAG] killed ray daemons referencing $RAY_PAT (self/ancestors excluded)"
    fi
else
    echo "[$TAG] no ray daemons referencing $RAY_PAT"
fi

# --- 2. Kill the scoped PID set, SIGTERM then SIGKILL ---------------------
if [[ -n "$FINAL_KILL" ]]; then
    echo "[$TAG] kill set ($(echo "$FINAL_KILL" | wc -l) pids):"
    # pid + basename only, never the full cmdline (it can carry tokens).
    for pid in $FINAL_KILL; do
        bn=$(ps -o comm= -p "$pid" 2>/dev/null | tr -d ' ' || echo '?')
        echo "[$TAG]   pid=$pid comm=$bn"
    done
    for pid in $FINAL_KILL; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 3
    for pid in $FINAL_KILL; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
else
    echo "[$TAG] kill set is empty (no in-scope ray/vllm/verl processes)"
fi

# --- 3. Unset stale torch distributed env vars ----------------------------
# verl re-derives MASTER_ADDR/PORT per cluster, but values inherited from the
# parent shell leak into the next ray.init runtime env and confuse rendezvous.
for v in MASTER_ADDR MASTER_PORT RANK LOCAL_RANK WORLD_SIZE \
         TORCHELASTIC_USE_AGENT_STORE RAY_ADDRESS; do
    if [[ -n "${!v:-}" ]]; then
        echo "[$TAG] unsetting $v (was: ${!v})"
        unset "$v"
    fi
done

# --- 4. Wipe ray session dirs (these hold AF_UNIX sockets + locks) -------
# RAY_TMPDIR (set in env_prelude.sh) is our *own* session dir, never another
# ray cluster's.
RAY_DIR=${RAY_TMPDIR:-/tmp/ray_rzero}
if [[ -d "$RAY_DIR" ]]; then
    find "$RAY_DIR" -mindepth 1 -maxdepth 1 -name 'session_*' -exec rm -rf {} + 2>/dev/null || true
    rm -f "$RAY_DIR/ray_current_cluster" 2>/dev/null || true
    echo "[$TAG] cleaned $RAY_DIR/session_*"
fi

# --- 5. Sleep so the kernel releases TIME_WAIT sockets + GPU memory ------
# 30s is enough for slow ray head daemons and the vLLM EngineCore subprocesses
# to release their bound TCP/IPC ports.
sleep 30

# --- 6. Observability dump: surface any survivor processes -----------------
# Re-run the seed collection so in-scope GPU-context holders that appeared
# after step 2 are reported too.
NEW_SEED=$(collect_seed_pids)
if [[ -n "$NEW_SEED" ]]; then
    echo "[$TAG] WARNING: survivor GPU-context holders on RZERO_GPUS:"
    for pid in $NEW_SEED; do
        bn=$(ps -o comm= -p "$pid" 2>/dev/null | tr -d ' ' || echo '?')
        echo "[$TAG]   pid=$pid comm=$bn"
    done
else
    echo "[$TAG] no survivor GPU-context holders on RZERO_GPUS"
fi
echo "[$TAG] cleanup done"
