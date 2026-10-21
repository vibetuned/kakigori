# Stage runner shared by pipeline-lite.sh and pipeline-gnn.sh (sourced, run
# from the repo root, with OUT set). Each stage is a training command that
# creates one run_NNN under $OUT; it runs under train.sh (which resumes after
# a crash) and leaves markers, so rerunning a pipeline after a reboot skips
# finished stages and resumes an interrupted one.

export PATH="$HOME/.local/bin:$PATH"
# the progress bar every 30 s instead of 10x a second: the log stays tail-able
export TQDM_MININTERVAL=30

say() { echo "[$(date +'%F %T')] $*"; }

latest_run() { ls -d "$OUT"/run_* 2>/dev/null | sort | tail -1; }

# The run dir a finished stage produced
stage_run() { cat "$OUT/.done_$1"; }

# run_stage <name> <training command...>
run_stage() {
    local name=$1
    shift
    if [ -f "$OUT/.done_$name" ]; then
        say "stage $name already done: $(stage_run "$name")"
        return 0
    fi
    local resume=()
    if [ -f "$OUT/.started_$name" ]; then
        # resume this stage's own run, never a finished stage's (interrupted
        # before the trainer created its run dir: start the stage afresh)
        if cat "$OUT"/.done_* 2>/dev/null | grep -qx "$(latest_run)"; then
            say "stage $name was interrupted before its run started: starting afresh"
        else
            resume=(--resume)
            say "stage $name was interrupted: resuming $(latest_run)"
        fi
    fi
    say "===== stage $name: $* ====="
    touch "$OUT/.started_$name"
    # progress = a new checkpoint under $OUT (train.sh tells bugs from faults)
    if ! PROGRESS_DIR="$OUT" ./train.sh "$@" "${resume[@]}"; then
        say "stage $name FAILED"
        exit 1
    fi
    latest_run > "$OUT/.done_$name"
    say "stage $name done: $(stage_run "$name")"
}

# eval_detector <label> <checkpoint>: eval-model on $VAL (validation-lite by
# default; a detector checkpoint, or the detector inside a GNN / joint wrapper)
eval_detector() {
    local eval_log="$OUT/eval_$1.log" val=${VAL:-data/validation-lite}
    say "eval $1: $2 -> $eval_log"
    for attempt in 1 2 3; do
        # a class whose mAP worker crashes is silently scored 0: retry
        uv run eval-model --checkpoint "$2" \
            --img-dir "$val/imgs" --ann-dir "$val/annotations" \
            --config conf-lite/config.json --hierarchy conf-lite/hierarchy.json \
            --tb-dir "$OUT/eval_tb" > "$eval_log" 2>&1
        grep -q "mAP @" "$eval_log" && ! grep -q "\[!\]" "$eval_log" && break
        say "eval $1 attempt $attempt failed or had a crashed class, see $eval_log"
    done
    grep "INFO - mAP @" "$eval_log" | sed "s/^.* - INFO - /    $1 /"
}
