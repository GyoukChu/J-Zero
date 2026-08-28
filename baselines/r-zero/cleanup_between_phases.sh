#!/bin/bash
# Cleanup of leftover ray cluster / verl trainer / vLLM workers between phases,
# so the next `verl.trainer.main_ppo` starts a *fresh* ray cluster instead of
# colliding with stale TCPStore endpoints and dashboard ports. Killing only
# `ray::` workers is not enough: ray's head daemons (gcs_server, raylet,
# dashboard_agent, ...) are what hold the ports the new cluster races for.
#
# *** SCOPING ***
# Never use a global `pkill -KILL -f '<pattern>'` here — it matches any process
# on the host, including other users' / other runs' vLLM workers. Kills are
# restricted to PIDs holding a CUDA context on RZERO_GPUS (default 0,1,2,3) plus
# their descendants and pattern-matching ancestors. PIDs on GPUs outside
# RZERO_GPUS are *never* killed.
#
# Usage: bash cleanup_between_phases.sh [tag]
#   tag: free-form string for log lines (e.g. "post-challenger", "solver-entry").
set -u

TAG=${1:-"cleanup"}
echo "[$TAG] starting scoped cleanup"

# --- 0. Define the GPU scope we are allowed to touch ----------------------
# Default mirrors the recipe: challenger uses 0,1 (trainer) + 2,3 (reward vLLM
# servers), solver uses 0-3, so 0-3 covers every phase.
RZERO_GPUS="${RZERO_GPUS:-0,1,2,3}"
echo "[$TAG] scope: GPUs $RZERO_GPUS"

# Seed set: PIDs currently holding a CUDA context on RZERO_GPUS, per nvidia-smi.
# Expanded with descendants and ancestors below.
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

# Expand a seed PID set with the descendants of those PIDs, iterating `pgrep -P`
# to a fixed point.
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

# Ancestors are only ever added when their cmdline matches one of these, so an
# unrelated daemon can never be swept up by the PPID walk.
PATTERNS_RX='ray::|VLLM::EngineCore|start_vllm_server\.py|verl\.trainer\.main_ppo|verl\.model_merger|gcs_server|raylet|dashboard_agent\.py|log_monitor\.py|monitor\.py.*--logs-dir|default_worker\.py|multiprocessing\.resource_tracker'

# One PPID hop, gated by PATTERNS_RX: the parent of a VLLM::EngineCore (e.g.
# python -m verl.trainer.main_ppo) is CPU-only and never appears in nvidia-smi.
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

# Pattern-matching PIDs catch e.g. a verl.trainer.main_ppo whose vLLM child was
# already killed, so it holds no GPU context of its own.
PATTERN_PIDS=$(pgrep -f "$PATTERNS_RX" 2>/dev/null | sort -u || true)

# Keep only PIDs whose ancestor chain intersects SEED; without that test we
# would match unrelated vLLM processes on out-of-scope GPUs.
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

INITIAL=$(printf '%s\n%s' "$SEED" "$PATTERN_IN_SCOPE" | sed '/^$/d' | sort -u)
if [[ -z "$INITIAL" ]]; then
    KILL_SET=""
else
    KILL_SET=$(echo "$INITIAL" | expand_descendants | add_matching_ancestors)
fi

# Set RZERO_KILL_EXCLUDE_RX when co-running another experiment on this node:
# nvidia-smi can transiently mis-attribute a foreign pid to our GPU set, and
# sweeping it kills that run's vLLM engine. Engine procs carry no project name
# in argv, hence the ancestor walk.
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

# Hard-exclude our own shell + its ancestors (just in case).
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
# Never `ray stop --force` here: it matches ray component process names for the
# whole user account regardless of temp dir, killing an unrelated concurrent
# run's raylet mid-training. Ray daemons embed their session path in argv, so
# matching on RAY_TMPDIR hits exactly this run's cluster.
RAY_PAT=${RAY_TMPDIR:-/tmp/ray_rzero}
# A bare pkill -f would also match our own ancestor shells whose command line
# mentions $RAY_TMPDIR, killing the caller mid-run — hence the self/ancestor skip.
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
    # Print pid + basename only (no full cmdline -> never leaks tokens).
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
# parent shell would reach the next ray.init's runtime env and confuse rendezvous.
for v in MASTER_ADDR MASTER_PORT RANK LOCAL_RANK WORLD_SIZE \
         TORCHELASTIC_USE_AGENT_STORE RAY_ADDRESS; do
    if [[ -n "${!v:-}" ]]; then
        echo "[$TAG] unsetting $v (was: ${!v})"
        unset "$v"
    fi
done

# --- 4. Wipe ray session dirs (these hold AF_UNIX sockets + locks) -------
# RAY_TMPDIR (set by env_prelude.sh) is our own session dir, never another run's.
RAY_DIR=${RAY_TMPDIR:-/tmp/ray_rzero}
if [[ -d "$RAY_DIR" ]]; then
    find "$RAY_DIR" -mindepth 1 -maxdepth 1 -name 'session_*' -exec rm -rf {} + 2>/dev/null || true
    rm -f "$RAY_DIR/ray_current_cluster" 2>/dev/null || true
    echo "[$TAG] cleaned $RAY_DIR/session_*"
fi

# --- 5. Sleep so the kernel releases TIME_WAIT sockets + GPU memory ------
# 30s covers slow ray head daemons releasing bound TCP ports and vLLM
# EngineCore subprocesses letting go of their ephemeral IPC ports.
sleep 30

# --- 6. Observability dump: surface any survivor processes -----------------
# Re-run the seed collection rather than a global pgrep: in-scope only, and it
# also catches GPU-context holders that appeared since step 2.
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
