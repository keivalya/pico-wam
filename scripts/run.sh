#!/usr/bin/env bash
# End-to-end pico-WAM experiment, repeated over several seeds:
#   for each seed: data -> train WAM -> train mini-VLA baseline -> evaluate -> closed-loop test
#                  (Push-T: also the WAM planning with its world model)
#   then: visualize the first seed, Push-T extras (linear-model baseline, figure
#   data, compute costs), and summarize mean ± std over seeds
# MODE=supervision runs the matched-supervision pilot instead (scripts/supervision_ablation.py).
#
# Usage: bash scripts/run.sh                (from the repo root; Push-T)
#        bash scripts/run_toy.sh            (the paper's point-mass study)
# Outputs land in $OUT/seed*/ (default results/<env>/), $VIS_DIR/ and videos/
# inside the repo. $OUT is deleted when the run starts.
set -euo pipefail

# ======================= experiment configuration =======================
# Each experiment in the study changes one of these lines (and nothing else).
ENV_NAME="${ENV_NAME:-pusht}"            # pusht or toy
PUSHT_DATA="${PUSHT_DATA:-lerobot}"      # pusht: lerobot (human demos), expert (scripted, with random bursts) or mixed (both)
DEMOS="${DEMOS:-206}"                    # pusht: how many of LeRobot's 206 human demos to use (the first N)
MODE="${MODE:-pipeline}"                 # pipeline, or supervision (the matched-supervision pilot)
PLAN="${PLAN:-1}"                        # pusht: also test the WAM planning with its world model (scripts/plan.py)
PLAN_COMPARE="${PLAN_COMPARE:-1}"        # pusht: also test the old planner, whose search starts from the last plan (warm_start)
PLAN_SWEEP="${PLAN_SWEEP:-0}"            # pusht: 1 = also test bigger search budgets: wider noise, 3 rounds, no policy prior
BRANCHES="${BRANCHES:-1}"                # pusht: 1 = the WAM also learns from simulator branches of the demos (scripts/branch_demos.py)
BRANCH_LENGTH="${BRANCH_LENGTH:-16}"      # pusht: actions per branch; training windows (HORIZON) only use branches at least this long
PARALLEL_SEEDS="${PARALLEL_SEEDS:-1}"    # 1 = run all seeds at once on the job's GPU (same results, less wall time)
case "$ENV_NAME" in
  pusht) DEFAULT_EPISODES=200; DEFAULT_MAX_STEPS=300; DEFAULT_RANDOM_ACTION_PROB=0.1 ;;
  toy)   DEFAULT_EPISODES=400; DEFAULT_MAX_STEPS=80;  DEFAULT_RANDOM_ACTION_PROB=0.3 ;;
  *) echo "[run] unknown ENV_NAME=$ENV_NAME (use pusht or toy)"; exit 1 ;;
esac
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-100}"
EPISODES="${EPISODES:-$DEFAULT_EPISODES}"     # collected demonstrations per seed (90% train / 10% unseen)
MAX_STEPS="${MAX_STEPS:-$DEFAULT_MAX_STEPS}"  # per collected and tested episode
RANDOM_ACTION_PROB="${RANDOM_ACTION_PROB:-$DEFAULT_RANDOM_ACTION_PROB}"   # exploration bursts in the data
HORIZON="${HORIZON:-15}"                 # imagination steps per training window
STATE_WEIGHT="${STATE_WEIGHT:-0.5}"
PREDICT_IMAGE="${PREDICT_IMAGE:-1}"      # 1 = the WAM also decodes images, so you can watch its imagination; 0 trains about 5x faster
IMAGE_WEIGHT="${IMAGE_WEIGHT:-1.0}"
IMAGE_FOCUS="${IMAGE_FOCUS:-0}"          # image loss: pixels that move, the T and the agent, count 1 + IMAGE_FOCUS times
LATENT_WEIGHT="${LATENT_WEIGHT:-1.0}"
STATE_INPUT="${STATE_INPUT:-1}"          # 0 = image + language only (no state_t input)
ACTION_HEAD="${ACTION_HEAD:-mlp}"        # mlp, diffusion or flow
ACTION_CHUNK="${ACTION_CHUNK:-8}"        # predict the next K actions at once and execute K/2 (1 = one action per step)
TEMPORAL_ENSEMBLE="${TEMPORAL_ENSEMBLE:-0}"  # 1 = predict a chunk every step and average the chunks' guesses, as in ACT
POLICY_HIDDEN="${POLICY_HIDDEN:-512}"    # action head: width of its hidden layers, for the WAM and mini-VLA alike
POLICY_LAYERS="${POLICY_LAYERS:-3}"      # action head: number of hidden layers
D_MODEL="${D_MODEL:-128}"                # width of the shared latent z; the dynamics MLP is twice as wide
IMAGE_ENCODER="${IMAGE_ENCODER:-keypoints}"  # keypoints = CNN + spatial-softmax keypoints; cnn = CNN + global average pooling
LR_SCHEDULE="${LR_SCHEDULE:-constant}"   # constant, or cosine decay to 0 (untested in the study)
TEST_EPISODES="${TEST_EPISODES:-100}"
DEVICE="${DEVICE:-auto}"
OUT="${OUT:-results/$ENV_NAME}"
VIS_DIR="${VIS_DIR:-outputs/$ENV_NAME}"
# ========================================================================

# ---- environment: one venv per package list, reused across runs
#      (set PICO_WAM_PY=/path/to/python to use an existing environment instead)
PKGS='torch numpy opencv-python-headless imageio imageio[ffmpeg] matplotlib pymunk huggingface_hub pyarrow av'
PY="${PYTHON:-python3}"
export PATH="$HOME/.local/bin:$PATH"     # batch jobs may not see a user-installed uv otherwise

if command -v module >/dev/null 2>&1 && [ -n "${PICO_WAM_MODULES:-}" ]; then
  module load $PICO_WAM_MODULES
fi

# CPU-only torch wheels unless this machine has an NVIDIA GPU (much smaller install)
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  TORCH_INDEX="https://download.pytorch.org/whl/cu126"
else
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi

if [ -n "${PICO_WAM_PY:-}" ]; then
  PYV="$PICO_WAM_PY"
else
  if command -v uv >/dev/null 2>&1; then
    PY_SPEC="${UV_PYTHON_VERSION:-3.11}"      # uv fetches this Python if needed
  else
    PY_SPEC="$($PY --version 2>&1)"
  fi
  PKG_HASH="$(echo "$PKGS $PY_SPEC $TORCH_INDEX" | cksum | cut -d' ' -f1)"
  VENV="$HOME/.cache/pico-wam/venv-$PKG_HASH"
  # .ready is written only after a complete install, so an interrupted
  # install is rebuilt instead of silently reused
  if [ ! -f "$VENV/.ready" ]; then
    rm -rf "$VENV"
    echo "[run] creating venv $VENV (python $PY_SPEC, torch from $TORCH_INDEX)"
    mkdir -p "$(dirname "$VENV")"
    if command -v uv >/dev/null 2>&1; then
      uv venv "$VENV" --python "$PY_SPEC"
      VIRTUAL_ENV="$VENV" uv pip install torch --index-url "$TORCH_INDEX"
      VIRTUAL_ENV="$VENV" uv pip install $PKGS
    else
      "$PY" -m venv "$VENV"
      "$VENV/bin/pip" install --upgrade pip
      "$VENV/bin/pip" install torch --index-url "$TORCH_INDEX"
      "$VENV/bin/pip" install $PKGS
    fi
    touch "$VENV/.ready"
  fi
  PYV="$VENV/bin/python"
fi
export PYTHONUNBUFFERED=1

echo "[run] host=$(hostname) python=$($PYV --version) torch=$($PYV -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")')"
echo "[run] cpus: $(nproc) slurm_job=${SLURM_JOB_ID:-none}"
echo "[run] config: ENV_NAME=$ENV_NAME PUSHT_DATA=$PUSHT_DATA DEMOS=$DEMOS MODE=$MODE PLAN=$PLAN PLAN_COMPARE=$PLAN_COMPARE PLAN_SWEEP=$PLAN_SWEEP BRANCHES=$BRANCHES BRANCH_LENGTH=$BRANCH_LENGTH SEEDS='$SEEDS' EPOCHS=$EPOCHS EPISODES=$EPISODES MAX_STEPS=$MAX_STEPS RANDOM_ACTION_PROB=$RANDOM_ACTION_PROB HORIZON=$HORIZON ACTION_CHUNK=$ACTION_CHUNK TEMPORAL_ENSEMBLE=$TEMPORAL_ENSEMBLE STATE_WEIGHT=$STATE_WEIGHT PREDICT_IMAGE=$PREDICT_IMAGE IMAGE_WEIGHT=$IMAGE_WEIGHT IMAGE_FOCUS=$IMAGE_FOCUS LATENT_WEIGHT=$LATENT_WEIGHT STATE_INPUT=$STATE_INPUT ACTION_HEAD=$ACTION_HEAD POLICY_HIDDEN=$POLICY_HIDDEN POLICY_LAYERS=$POLICY_LAYERS D_MODEL=$D_MODEL IMAGE_ENCODER=$IMAGE_ENCODER LR_SCHEDULE=$LR_SCHEDULE TEST_EPISODES=$TEST_EPISODES"

STATE_FLAG=""
[ "$STATE_INPUT" = "0" ] && STATE_FLAG="--no_state_input"
IMAGE_FLAG=""
[ "$PREDICT_IMAGE" = "1" ] && IMAGE_FLAG="--predict_image"
ENSEMBLE_FLAG=""
[ "$TEMPORAL_ENSEMBLE" = "1" ] && ENSEMBLE_FLAG="--temporal-ensemble"
MODEL_FLAGS="--action_head $ACTION_HEAD --action-chunk $ACTION_CHUNK --policy-hidden $POLICY_HIDDEN --policy-layers $POLICY_LAYERS --d-model $D_MODEL --image_encoder $IMAGE_ENCODER --lr-schedule $LR_SCHEDULE"
TORCH_DEVICE="$DEVICE"   # scripts that take a torch device name, not "auto"
[ "$DEVICE" = "auto" ] && TORCH_DEVICE="$($PYV -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"

# 1. data: LeRobot's human demos (the same for every seed; the seed changes the
#    train/val split and the initialization), a fresh scripted collection per
#    seed, or both merged
get_data() {  # get_data <dir> <seed>
  local R="$1" SEED="$2" DEMO_FLAG=""
  [ "$DEMOS" != "206" ] && DEMO_FLAG="--episodes $DEMOS"
  if [ "$ENV_NAME" = "pusht" ] && [ "$PUSHT_DATA" != "expert" ]; then
    $PYV -m scripts.download_pusht $DEMO_FLAG --output-path "$R/lerobot.npz" | tail -7
  fi
  if [ "$ENV_NAME" = "toy" ] || [ "$PUSHT_DATA" != "lerobot" ]; then
    $PYV -m scripts.collect_data --env "$ENV_NAME" --episodes "$EPISODES" --max-steps "$MAX_STEPS" \
      --seed $((42 + 1000 * SEED)) --random-action-prob "$RANDOM_ACTION_PROB" \
      --output-path "$R/collected.npz" | tail -3
  fi
  if [ -f "$R/lerobot.npz" ] && [ -f "$R/collected.npz" ]; then
    $PYV -m scripts.merge_datasets "$R/lerobot.npz" "$R/collected.npz" --output-path "$R/$ENV_NAME.npz"
    rm "$R/lerobot.npz" "$R/collected.npz"
  else
    mv "$R"/lerobot.npz "$R/$ENV_NAME.npz" 2>/dev/null || mv "$R/collected.npz" "$R/$ENV_NAME.npz"
  fi
}

# for_each_seed <function>: run "<function> <seed>" for every seed, one after
# another, or with PARALLEL_SEEDS=1 all at once, each logging to its own file;
# the logs are printed in seed order once every seed has finished
for_each_seed() {
  if [ "$PARALLEL_SEEDS" = "1" ]; then
    local SEED pid status=0 pids=()
    for SEED in $SEEDS; do
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$1" "$SEED" > "$OUT/log-seed$SEED.txt" 2>&1 &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    for SEED in $SEEDS; do cat "$OUT/log-seed$SEED.txt"; done
    return $status
  fi
  local SEED
  for SEED in $SEEDS; do "$1" "$SEED"; done
}

rm -rf "$OUT"
mkdir -p "$OUT"

if [ "$MODE" = "supervision" ]; then
  pilot_seed() {
    local SEED="$1"
    R="$OUT/seed-$SEED"
    mkdir -p "$R"
    echo "[run] ===== supervision pilot, seed $SEED ====="
    get_data "$R" "$SEED"
    $PYV -m scripts.supervision_ablation --dataset "$R/$ENV_NAME.npz" --seed "$SEED" --epochs "$EPOCHS" \
      --device "$TORCH_DEVICE" --output "$R/ablation" | grep -E "RESULT"
  }
  for_each_seed pilot_seed
  $PYV -m scripts.summarize_supervision --root "$OUT" --seeds "$(echo $SEEDS | wc -w)"
  echo "[run] done"
  exit 0
fi

run_seed() {
  local SEED="$1"
  R="$OUT/seed$SEED"
  mkdir -p "$R"
  echo "[run] ===== seed $SEED ====="
  get_data "$R" "$SEED"

  # the world model also learns from simulator branches of the demos ("what if
  # the human had pushed differently?"); mini-VLA has no world model, so it keeps
  # training on the demos alone
  WAM_DATA="$R/$ENV_NAME.npz"
  if [ "$ENV_NAME" = "pusht" ] && [ "$BRANCHES" = "1" ]; then
    $PYV -m scripts.branch_demos "$R/$ENV_NAME.npz" --seed "$SEED" --length "$BRANCH_LENGTH" --output-path "$R/branched.npz" | tail -1
    WAM_DATA="$R/branched.npz"
  fi

  # 2. WAM and the mini-VLA baseline (same demos, same action head, same seed)
  $PYV -m scripts.train --dataset-path "$WAM_DATA" --model_type wam $MODEL_FLAGS \
    --predict_state $IMAGE_FLAG $STATE_FLAG --horizon "$HORIZON" \
    --state-weight "$STATE_WEIGHT" --image-weight "$IMAGE_WEIGHT" --image-focus "$IMAGE_FOCUS" --latent-weight "$LATENT_WEIGHT" \
    --epochs "$EPOCHS" --seed "$SEED" --device "$DEVICE" --save-path "$R/wam.pt" | grep -E "Model:|Episodes:|Epoch $EPOCHS/| val:" | tail -4

  $PYV -m scripts.train --dataset-path "$R/$ENV_NAME.npz" --model_type vla $MODEL_FLAGS \
    $STATE_FLAG --epochs "$EPOCHS" --seed "$SEED" --device "$DEVICE" --save-path "$R/vla.pt" | grep -E "Model:|Epoch $EPOCHS/| val:" | tail -3

  # 3. world-model evaluation on unseen episodes (+ counterfactuals)
  $PYV -m scripts.evaluate --checkpoint "$R/wam.pt" --max-horizon 15 --device "$DEVICE" --output "$R/wam_eval.json"

  # 4. closed loop on unseen env seeds
  $PYV -m scripts.test --checkpoint "$R/wam.pt" --episodes "$TEST_EPISODES" --seed $((10000 + 1000 * SEED)) \
    --max-steps "$MAX_STEPS" --device cpu $ENSEMBLE_FLAG --json-out "$R/wam_test.json" | tail -1
  $PYV -m scripts.test --checkpoint "$R/vla.pt" --episodes "$TEST_EPISODES" --seed $((10000 + 1000 * SEED)) \
    --max-steps "$MAX_STEPS" --device cpu $ENSEMBLE_FLAG --json-out "$R/vla_test.json" | tail -1
  if [ "$ENV_NAME" = "pusht" ] && [ "$PLAN" = "1" ]; then
    $PYV -m scripts.test --checkpoint "$R/wam.pt" --planner mpc --episodes "$TEST_EPISODES" --seed $((10000 + 1000 * SEED)) \
      --max-steps "$MAX_STEPS" --device "$TORCH_DEVICE" --json-out "$R/wam_mpc_test.json" | tail -1
    if [ "$PLAN_COMPARE" = "1" ]; then   # the old planner: each search starts from the last step's plan
      $PYV -m scripts.test --checkpoint "$R/wam.pt" --planner mpc --planner-config '{"warm_start": true}' \
        --episodes "$TEST_EPISODES" --seed $((10000 + 1000 * SEED)) \
        --max-steps "$MAX_STEPS" --device "$TORCH_DEVICE" --json-out "$R/wam_mpc_warm_test.json" | tail -1
    fi
  fi
  if [ "$ENV_NAME" = "pusht" ] && [ "$PLAN_SWEEP" = "1" ]; then
    # the same planner with more search (three cross-entropy rounds, wider noise), or without the policy's
    # prior, all warm-started like the study-base planner these budgets were measured with
    for CFG in 'cem {"iterations": 3, "noise": 0.3, "warm_start": true}' 'wide {"noise": 0.3, "warm_start": true}' \
               'noprior {"prior": "zero", "iterations": 3, "noise": 0.3, "warm_start": true}'; do
      NAME="${CFG%% *}"
      $PYV -m scripts.test --checkpoint "$R/wam.pt" --planner mpc --planner-config "${CFG#* }" --episodes "$TEST_EPISODES" \
        --seed $((10000 + 1000 * SEED)) --max-steps "$MAX_STEPS" --device "$TORCH_DEVICE" \
        --json-out "$R/wam_mpc_${NAME}_test.json" | tail -1 | sed "s/^/[$NAME] /"
    done
  fi
}
for_each_seed run_seed

# ---- 5. look inside the world model (first seed)
FIRST_SEED="${SEEDS%% *}"
VIS_FLAGS="--video"
# Push-T episodes are long: the README GIF shows the first 80 steps
[ "$ENV_NAME" = "pusht" ] && VIS_FLAGS="--video --gif --horizon 80"
$PYV -m scripts.visualize_rollout --checkpoint "$OUT/seed$FIRST_SEED/wam.pt" --out-dir "$VIS_DIR" $VIS_FLAGS --device "$DEVICE"

# ---- 6. Push-T extras: a linear-model baseline of the dynamics, the data behind
#         the paper's qualitative figures, and parameter and runtime costs (CPU)
if [ "$ENV_NAME" = "pusht" ]; then
  $PYV -m scripts.pusht_dynamics_audit --run-dir "$OUT" | tail -5
  $PYV docs/figures/export_figure_data.py --run-dir "$OUT" --name run --out "$OUT/figure-data" | grep -E "wrote" | wc -l | sed 's/^/[run] figure-data files: /'
  $PYV docs/figures/benchmark_efficiency.py --run-dir "$OUT/seed$FIRST_SEED" --out "$OUT/efficiency" > /dev/null && echo "[run] efficiency: $OUT/efficiency/efficiency.json"
fi

# ---- 7. mean ± std over seeds
$PYV -m scripts.summarize_runs --root "$OUT"

echo "[run] done"
