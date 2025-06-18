#!/bin/bash

GAME_NAME=${1:-"Alien"}  #
AGENT_NAME=${AGENT_NAME:-metric_rainbow}  # Default to metric_dqn_bper if no agent name is specified

seeds=(118398 919409 711872 442081 189061)
SLURM_ARRAY_TASK_ID=0

# Select the seed based on the SLURM array task ID
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}

python -m train \
    --base_dir=outputs/logs/ \
    --gin_files=rainbow.gin \
    --game_name=${GAME_NAME} \
    --agent_name=${AGENT_NAME} \
    --seed=${SEED}