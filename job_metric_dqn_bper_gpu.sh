#!/bin/bash
#SBATCH --job-name=bisimulation-rl-DQN
#SBATCH --array=0-2
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=o.v.guarnizocabezas@bham.ac.uk
#SBATCH --qos=bbgpu
#SBATCH --cpus-per-task=14
#SBATCH --account=giacobbm-bisimulation-rl
#SBATCH --gres=gpu:a30:1
#SBATCH --output="outputs/slurm-files/slurm-DQN-%A_%a.out"

module purge; module load bluebear
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0
module load cuDNN/8.9.2.26-CUDA-12.1.1


GAME_LIST=("$@")
# If no games are provided, default to a predefined list
if [ ${#GAME_LIST[@]} -eq 0 ]; then
    GAME_LIST=("AirRaid"
                "Alien" 
                "Amidar")
fi

AGENT_NAME=${AGENT_NAME:-metric_dqn_bper}  # Default to metric_dqn_bper if no agent name is specified
BPER_SCHEME=${BPER_SCHEME:-"scaling"}  # Default to softmax if no BPER scheme is specified

# Temporary scratch space for I/O efficiency
BB_WORKDIR=$(mktemp -d /scratch/${USER}_${SLURM_JOBID}.XXXXXX)
export TMPDIR=${BB_WORKDIR}


set -x  # Enable debug mode
set -e

# pip install torch==2.3.1 torchvision==0.18.1

PROJECT_DIR="/rds/projects/g/giacobbm-bisimulation-rl"
export VENV_DIR="${PROJECT_DIR}/virtual-environments"
export VENV_PATH="${VENV_DIR}/dopamine-gpu2-virtual-env-${BB_CPU}"

# Create a master venv directory if necessary
mkdir -p ${VENV_DIR}

# Check if virtual environment exists and create it if not
if [[ ! -d ${VENV_PATH} ]]; then
    python3 -m venv --system-site-packages ${VENV_PATH}
fi

# Activate the virtual environment
source ${VENV_PATH}/bin/activate

# Store pip cache in /scratch directory, instead of the default home directory location
PIP_CACHE_DIR="/scratch/${USER}/pip"


# Perform any required pip installations. For reasons of consistency we would recommend
# that you define the version of the Python module – this will also ensure that if the
# module is already installed in the virtual environment it won't be modified.
# python3 -m pip install --no-cache-dir --upgrade pip
# pip install dopamine-rl
# python3 -m pip install 'tensorflow[and-cuda]'
# cd baselines && pip install -e .
# cd ..
# pip install ale-py
# pip install seaborn
# pip install tqdm
# pip uninstall -y jax jaxlib
# pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


seeds=(118398 919409 711872) # 442081 189061)

SEED=${seeds[$SLURM_ARRAY_TASK_ID]}

# Execute each game sequentially
for GAME_NAME in "${GAME_LIST[@]}"; do
    echo "Game name: $GAME_NAME"
    echo "Agent name: $AGENT_NAME"

    # Record the start time for the current game
    game_start_time=$(date +%s)
    echo "Starting game $GAME_NAME at $(date)"

    # Execute based on the selected variant
    if [ "$AGENT_NAME" == "metric_dqn_bper" ]; then
        python -m train \
            --base_dir=logs/ \
            --gin_files=dqn.gin \
            --game_name=${GAME_NAME} \
            --agent_name="${AGENT_NAME}_${BPER_SCHEME}" \
            --seed=${SEED} \
            --gin_bindings="MetricDQNBPERAgent.method_scheme='${BPER_SCHEME}'"
    elif [ "$AGENT_NAME" == "metric_dqn_per" ]; then
        python -m train \
            --base_dir=logs/ \
            --gin_files=dqn.gin \
            --game_name=${GAME_NAME} \
            --agent_name=${AGENT_NAME} \
            --seed=${SEED} \
            --gin_bindings="MetricDQNBPERAgent.bper_weight=0"
    elif [ "$AGENT_NAME" == "metric_dqn" ]; then
        python -m train \
            --base_dir=logs/ \
            --gin_files=dqn.gin \
            --game_name=${GAME_NAME} \
            --agent_name=${AGENT_NAME} \
            --seed=${SEED}
    fi

    # Calculate runtime for the current game
    game_end_time=$(date +%s)
    game_runtime=$((game_end_time - game_start_time))
    days=$((game_runtime / 86400))
    hours=$(( (game_runtime % 86400) / 3600 ))
    minutes=$(( (game_runtime % 3600) / 60 ))

    # Send notification for the current game
    {
        echo "Game completed at $(date)"
        echo "Game: $GAME_NAME"
        echo "Agent: $AGENT_NAME"
        echo "Seed: $SEED"
        echo "Runtime: ${days}d ${hours}h ${minutes}m"
        echo ""
        echo "SLURM Job ID: $SLURM_JOBID"
    } | mailx -s "Slurm Array Job: [Game Completed] ${AGENT_NAME} on ${GAME_NAME} (Seed: ${SEED})" o.v.guarnizocabezas@bham.ac.uk

    echo "Completed game $GAME_NAME at $(date)"
done

# Cleanup
test -d ${BB_WORKDIR} && /bin/rm -rf ${BB_WORKDIR}