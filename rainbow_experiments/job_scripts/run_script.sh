#!/bin/bash

GAME_NAME=${1:-"Alien"}  #
AGENT_NAME=${AGENT_NAME:-metric_rainbow}  # Default to metric_dqn_bper if no agent name is specified
BPER_SCHEME=${BPER_SCHEME:-"scaling"}  # Default to softmax if no BPER scheme is specified

seeds=(118398 919409 711872 442081 189061)
SLURM_ARRAY_TASK_ID=0

# Select the seed based on the SLURM array task ID
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}

if [ "$AGENT_NAME" == "metric_rainbow" ]; then
    echo "Running Metric Rainbow with PER"
    python -m train \
        --base_dir=outputs/logs/ \
        --gin_files=rainbow.gin \
        --game_name=${GAME_NAME} \
        --agent_name=${AGENT_NAME} \
        --seed=${SEED}
elif [ "$AGENT_NAME" == "metric_rainbow_bper" ]; then
    echo "Running Metric Rainbow with BPER SCHEME: ${BPER_SCHEME}"
    python -m train \
        --base_dir=outputs/logs/ \
        --gin_files=rainbow.gin \
        --game_name=${GAME_NAME} \
        --agent_name="${AGENT_NAME}_${BPER_SCHEME}" \
        --seed=${SEED} \
        --gin_bindings="MetricRainbowBPERAgent.bper_weight=1" \
        --gin_bindings="MetricRainbowBPERAgent.method_scheme='${BPER_SCHEME}'"
fi

