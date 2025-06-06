


# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
# Note that to run this you need to obtain the gin files for Dopamine JAX
# agents:
# DQN: github.com/google/dopamine/tree/master/dopamine/jax/agents/dqn/configs
# Rainbow: github.com/google/dopamine/tree/master/dopamine/jax/agents/rainbow/configs
# QR-DQN: github.com/google/dopamine/tree/master/dopamine/jax/agents/quantile/configs
# IQN: github.com/google/dopamine/tree/master/dopamine/jax/agents/implicit_quantile/configs
set -e
set -x

# virtualenv -p python3 .
# source ./bin/activate

# pip install -r mico/requirements.txt
GAME_NAME=${1:-"Alien"}  #
AGENT_NAME=${AGENT_NAME:-metric_dqn_bper}  # Default to metric_dqn_bper if no agent name is specified
BPER_SCHEME=${BPER_SCHEME:-"scaling"}  # Default to softmax if no BPER scheme is specified

seeds=(118398 919409 711872 442081 189061)
SLURM_ARRAY_TASK_ID=0

# Select the seed based on the SLURM array task ID
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}

# Execute based on the selected variant
if [ "$AGENT_NAME" == "metric_dqn_bper" ]; then
    python -m calculate_grads_norm \
        --base_dir=logs/ \
        --checkpoint_dir="logs/${GAME_NAME}/${AGENT_NAME}_${BPER_SCHEME}/${SEED}/" \
        --replay_buffer_ckpt_dir="logs/${GAME_NAME}/metric_dqn/118398/" \
        --gin_files=eval_metric_dqn.gin \
        --game_name=${GAME_NAME} \
        --agent_name="${AGENT_NAME}_${BPER_SCHEME}" \
        --seed=${SEED} \
        --gin_bindings="MetricDQNBPERAgent.method_scheme='${BPER_SCHEME}'"

    python -m calculate_post_train_stats \
        --base_dir=logs/ \
        --checkpoint_dir="logs/${GAME_NAME}/${AGENT_NAME}_${BPER_SCHEME}/${SEED}/" \
        --replay_buffer_ckpt_dir="logs/${GAME_NAME}/metric_dqn/118398/" \
        --gin_files=eval_metric_dqn.gin \
        --game_name=${GAME_NAME} \
        --agent_name="${AGENT_NAME}_${BPER_SCHEME}" \
        --seed=${SEED} \
        --gin_bindings="MetricDQNBPERAgent.method_scheme='${BPER_SCHEME}'"

    python -m calculate_exploration \
        --base_dir=logs/ \
        --checkpoint_dir="logs/${GAME_NAME}/${AGENT_NAME}_${BPER_SCHEME}/${SEED}/" \
        --replay_buffer_ckpt_dir="logs/${GAME_NAME}/metric_dqn/118398/" \
        --gin_files=eval_metric_dqn.gin \
        --game_name=${GAME_NAME} \
        --agent_name="${AGENT_NAME}_${BPER_SCHEME}" \
        --seed=${SEED} \
        --gin_bindings="MetricDQNBPERAgent.method_scheme='${BPER_SCHEME}'"

elif [ "$AGENT_NAME" == "metric_dqn_per" ]; then
    python -m calculate_grads_norm \
        --base_dir=logs/ \
        --checkpoint_dir="logs/${GAME_NAME}/${AGENT_NAME}/${SEED}/" \
        --replay_buffer_ckpt_dir="logs/${GAME_NAME}/metric_dqn/118398/" \
        --gin_files=eval_metric_dqn.gin \
        --game_name=${GAME_NAME} \
        --agent_name=${AGENT_NAME} \
        --seed=${SEED} \
        --gin_bindings="MetricDQNBPERAgent.bper_weight=0"

    python -m calculate_post_train_stats \
        --base_dir=logs/ \
        --checkpoint_dir="logs/${GAME_NAME}/${AGENT_NAME}/${SEED}/" \
        --replay_buffer_ckpt_dir="logs/${GAME_NAME}/metric_dqn/118398/" \
        --gin_files=eval_metric_dqn.gin \
        --game_name=${GAME_NAME} \
        --agent_name=${AGENT_NAME} \
        --seed=${SEED} \
        --gin_bindings="MetricDQNBPERAgent.bper_weight=0"

    python -m calculate_exploration \
        --base_dir=logs/ \
        --checkpoint_dir="logs/${GAME_NAME}/${AGENT_NAME}/${SEED}/" \
        --replay_buffer_ckpt_dir="logs/${GAME_NAME}/metric_dqn/118398/" \
        --gin_files=eval_metric_dqn.gin \
        --game_name=${GAME_NAME} \
        --agent_name=${AGENT_NAME} \
        --seed=${SEED} \
        --gin_bindings="MetricDQNBPERAgent.bper_weight=0"

elif [ "$AGENT_NAME" == "metric_dqn" ]; then
    python -m calculate_grads_norm \
        --base_dir=logs/ \
        --checkpoint_dir="logs/${GAME_NAME}/${AGENT_NAME}/${SEED}/" \
        --replay_buffer_ckpt_dir="logs/${GAME_NAME}/metric_dqn/118398/" \
        --gin_files=eval_metric_dqn.gin \
        --game_name=${GAME_NAME} \
        --agent_name=${AGENT_NAME} \
        --seed=${SEED} \
        --gin_bindings="MetricDQNBPERAgent.replay_scheme='uniform'"

    python -m calculate_post_train_stats \
        --base_dir=logs/ \
        --checkpoint_dir="logs/${GAME_NAME}/${AGENT_NAME}/${SEED}/" \
        --replay_buffer_ckpt_dir="logs/${GAME_NAME}/metric_dqn/118398/" \
        --gin_files=eval_metric_dqn.gin \
        --game_name=${GAME_NAME} \
        --agent_name=${AGENT_NAME} \
        --seed=${SEED} \
        --gin_bindings="MetricDQNBPERAgent.replay_scheme='uniform'"

    python -m calculate_exploration \
        --base_dir=logs/ \
        --checkpoint_dir="logs/${GAME_NAME}/${AGENT_NAME}/${SEED}/" \
        --replay_buffer_ckpt_dir="logs/${GAME_NAME}/metric_dqn/118398/" \
        --gin_files=eval_metric_dqn.gin \
        --game_name=${GAME_NAME} \
        --agent_name=${AGENT_NAME} \
        --seed=${SEED} \
        --gin_bindings="MetricDQNBPERAgent.replay_scheme='uniform'"

else
    echo "Unknown variant: $AGENT_NAME"
    exit 1
fi
