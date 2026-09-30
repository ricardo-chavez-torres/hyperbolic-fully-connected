#!/bin/bash
# Training-time benchmark behind the paper's runtime comparison: every model is trained for
# a few CIFAR-100 epochs, back to back, on the same GPU, so all ratios share one piece of
# hardware. Run from anywhere, on a machine with one CUDA GPU:
#
#   bash runtime_exp/benchmark_training_time.sh [NUM_EPOCHS] [OUT_DIR]
#
# NUM_EPOCHS (default 4) epochs per run; the first is discarded as warm-up/compilation and
# the median of the rest is the per-epoch cost. Each model runs eager and torch.compile'd:
#
#   Euclidean ResNet-18   cifar_exp/main.py --manifold euclidean
#   FGG-LNN ResNet-18     cifar_exp/main.py --lorentz_method ours   --norm_config centering_weightnorm
#   HCNN ResNet-18        cifar_exp/main.py --lorentz_method theirs --norm_config normal_noweightnorm
#   ILNN ResNet-18        baselines/ilnn/experiments/vision/train.py, config ILNN-CIFAR100.txt
#   Poincaré ResNet-20    runtime_exp/poincare_resnet_timing.py (model code in baselines/poincare_resnet)
#
# All runs use batch 128, 8 DataLoader workers and the same 45k/5k train/validation split
# (352 iterations/epoch). A data-loader-only pass gives the floor below which no model's
# epoch can go, and the eager Euclidean run is repeated at the end to expose drift (e.g.
# thermal throttling). With torch 2.9, ILNN and the Poincaré ResNet fail to compile (a Triton
# and an Inductor bug respectively); their compiled rows are then reported as failed.
#
# RUNS selects a subset (keep a Euclidean run in it: ratios are only taken within one run
# directory), e.g.  RUNS="euclidean_eager poincare_eager" bash runtime_exp/benchmark_training_time.sh
#
# Results: OUT_DIR (default runtime_exp/results/training_benchmark_<id>), one runtime.json
# record and log per run plus summary.txt. The records contain the hostname and absolute
# paths; runtime_exp/results/ is git-ignored for that reason.

set -uo pipefail

NUM_EPOCHS="${1:-4}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="$(mkdir -p "${2:-$REPO/runtime_exp/results/training_benchmark_$RUN_ID}" && cd "${2:-$REPO/runtime_exp/results/training_benchmark_$RUN_ID}" && pwd)"
DATA_DIR="$REPO/cifar_exp/data/cifar"   # shared by every model

export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export TQDM_MININTERVAL=60

echo "===== host=$(hostname)  epochs/run=$NUM_EPOCHS  |  $(date) ====="
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv,noheader || true
echo "OUT_DIR=$OUT_DIR"

# run NAME CMD...: one timed run, its output in $OUT_DIR/NAME.log, its record in $OUT_DIR/NAME.json
run() {
    local name=$1; shift
    echo
    echo ">>> $name  ($(date +%T))"
    timeout 90m "$@" > "$OUT_DIR/$name.log" 2>&1
    local status=$?
    echo "    exit=$status"
    grep -E "\[timing\] epoch|Error|error:" "$OUT_DIR/$name.log" | tail -n 8 | sed 's/^/    /'
    printf '%-22s exit=%s\n' "$name" "$status" >> "$OUT_DIR/STATUS.txt"
    nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks_throttle_reasons.active \
        --format=csv,noheader 2>/dev/null | sed 's/^/    gpu after run: /'
}

ours() {  # ours NAME [cifar_exp/main.py args...]
    local name=$1; shift
    run "$name" uv run --directory "$REPO/cifar_exp" python main.py \
        --dataset cifar100 --num_epochs "$NUM_EPOCHS" --num_workers 8 --seed 0 \
        --runtime_json "$OUT_DIR/$name.json" "$@"
}

ilnn() {  # ilnn NAME [train.py args...]; train.py writes runtime.json under its output dir
    local name=$1; shift
    local out="$OUT_DIR/ilnn_outputs/$name"
    mkdir -p "$out"   # train.py creates only the last path component (os.mkdir)
    run "$name" uv run --directory "$REPO" python baselines/ilnn/experiments/vision/train.py \
        -c "$REPO/baselines/ilnn/experiments/vision/config/ILNN-CIFAR100.txt" --device cuda:0 \
        --seed 1 --exp_name ILNN --num_epochs "$NUM_EPOCHS" --data_dir "$DATA_DIR" \
        --output_dir "$out" "$@"
    find "$out" -name runtime.json -exec cp {} "$OUT_DIR/$name.json" \; 2>/dev/null
}

poincare() {  # poincare NAME [driver args...]
    local name=$1; shift
    run "$name" uv run --directory "$REPO" python runtime_exp/poincare_resnet_timing.py \
        --num_epochs "$NUM_EPOCHS" --data_dir "$DATA_DIR" --runtime_json "$OUT_DIR/$name.json" "$@"
}

FGG="--lorentz_method ours --norm_config centering_weightnorm --exp_name FGG-LNN"
HCNN="--lorentz_method theirs --norm_config normal_noweightnorm --exp_name HCNN"
RUNS="${RUNS:-loader_only euclidean_eager euclidean_compiled fgglnn_eager fgglnn_compiled
    hcnn_eager hcnn_compiled ilnn_eager ilnn_compiled poincare_eager poincare_compiled
    euclidean_eager_end}"
for r in $RUNS; do
    case $r in
        loader_only)          poincare "$r" --loader_only --num_epochs 3 ;;
        euclidean_eager*)     ours "$r" --manifold euclidean --exp_name Euclidean ;;
        euclidean_compiled)   ours "$r" --manifold euclidean --exp_name Euclidean --compile ;;
        fgglnn_eager)         ours "$r" $FGG ;;
        fgglnn_compiled)      ours "$r" $FGG --compile ;;
        hcnn_eager)           ours "$r" $HCNN ;;
        hcnn_compiled)        ours "$r" $HCNN --compile ;;
        ilnn_eager)           ilnn "$r" ;;
        ilnn_compiled)        ilnn "$r" --compile ;;
        poincare_eager)       poincare "$r" ;;
        poincare_compiled)    poincare "$r" --compile ;;
        *)                    echo "unknown run: $r" ;;
    esac
done

echo
echo "===== summary ====="
uv run --directory "$REPO" python runtime_exp/summarize_training_benchmark.py "$OUT_DIR"
echo "===== DONE  |  $(date) ====="
