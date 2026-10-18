#!/usr/bin/env bash
# The paper's study on the 2D point-mass world: scripts/run.sh with the toy
# env, writing to results/seed*/ and outputs/wam/ as the study always has.
# Every setting of scripts/run.sh can still be changed, e.g.
#   RANDOM_ACTION_PROB=0.0 bash scripts/run_toy.sh
set -euo pipefail
# The point-mass study predates the Push-T defaults, so pin its own settings
# (one action per step, 15 epochs, image decoder on, the original action head
# with one hidden layer of 128) unless you override them.
ENV_NAME=toy OUT="${OUT:-results}" VIS_DIR="${VIS_DIR:-outputs/wam}" \
  ACTION_CHUNK="${ACTION_CHUNK:-1}" EPOCHS="${EPOCHS:-15}" PREDICT_IMAGE="${PREDICT_IMAGE:-1}" \
  POLICY_HIDDEN="${POLICY_HIDDEN:-128}" POLICY_LAYERS="${POLICY_LAYERS:-1}" \
  exec bash "$(dirname "$0")/run.sh"
