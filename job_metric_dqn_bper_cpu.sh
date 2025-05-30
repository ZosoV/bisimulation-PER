#!/bin/bash
#SBATCH --job-name=bisimulation-rl-DQN
#SBATCH --array=0-2
#SBATCH --ntasks=1
#SBATCH --time=10-00:00:00
#SBATCH --qos=bbdefault
#SBATCH --mail-type=ALL
#SBATCH --mail-user=o.v.guarnizocabezas@bham.ac.uk
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=8GB
#SBATCH --account=giacobbm-bisimulation-rl
#SBATCH --output="outputs/slurm-files/slurm-DQN-%A_%a.out"
#SBATCH --constraint=sapphire

module purge; module load bluebear
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0
module load tqdm/4.66.1-GCCcore-12.3.0
# module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
# module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
# module load bear-apps/2022a
# module load wandb/0.13.6-GCC-11.3.0

GAME_NAME=${1:-Alien}  # Default to Alien if no game name is specified
AGENT_NAME=${AGENT_NAME:-metric_dqn_bper}  # Default to metric_dqn_bper if no agent name is specified
# CUSTOM_THREADS=18
echo "Game name: $GAME_NAME"
echo "Agent name: $AGENT_NAME"
BPER_SCHEME=${BPER_SCHEME:-"scaling"}  # Default to softmax if no BPER scheme is specified

# Temporary scratch space for I/O efficiency
BB_WORKDIR=$(mktemp -d /scratch/${USER}_${SLURM_JOBID}.XXXXXX)
# BB_WORKDIR=$(mktemp -d /rds/projects/g/giacobbm-bisimulation-rl/${USER}_${SLURM_JOBID}.XXXXXX)
export TMPDIR=${BB_WORKDIR}
# export EXP_BUFF=${BB_WORKDIR}

set -x  # Enable debug mode
set -e

# pip install torch==2.3.1 torchvision==0.18.1

PROJECT_DIR="/rds/projects/g/giacobbm-bisimulation-rl"
export VENV_DIR="${PROJECT_DIR}/virtual-environments"
export VENV_PATH="${VENV_DIR}/dopamine-cpu-virtual-env-${BB_CPU}"

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


# NOTE: Only execute the first time
# Perform any required pip installations. For reasons of consistency we would recommend
# that you define the version of the Python module – this will also ensure that if the
# module is already installed in the virtual environment it won't be modified.
# pip install dopamine-rl
# cd baselines && pip install -e .
# cd ..
# pip install ale-py
# pip install seaborn

# NOTE: No jax cuda installation
# pip uninstall -y jax jaxlib
# pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

seeds=(118398 919409 711872 442081 189061)

# Select the seed based on the SLURM array task ID
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}

# Record the start time
start_time=$(date +%s)
echo "Starting task with seed $SEED at $(date)"

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
        --seed=${SEED} \
        --gin_bindings="MetricDQNBPERAgent.replay_scheme='uniform'"

fi

echo "Completed task with seed $SEED at $(date)"

exit_status=$?
end_time=$(date +%s)
runtime=$((end_time - start_time))
days=$((runtime / 86400))
hours=$(( (runtime % 86400) / 3600 ))
minutes=$(( (runtime % 3600) / 60 ))

slurm_output_file="outputs/slurm-files/slurm-DQN-${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}.out"

subject="Slurm Array Job CPU: ${AGENT_NAME} on ${GAME_NAME} (Seed: ${SEED})"
subject="${subject} [COMPLETED]"
{
    echo "Task finished at $(date)"
    echo "Seed: $SEED"
    echo "Game: $GAME_NAME"
    echo "Agent: $AGENT_NAME"
    echo "BPER Scheme: $BPER_SCHEME"
    echo "SLURM Job ID: $SLURM_JOB_ID"
    echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
    echo "Total runtime: ${days} days, ${hours} hours, ${minutes} minutes"
    echo ""
    cat "${PROJECT_DIR}/${slurm_output_file}" | grep Iteration
} | mailx -s "$subject" o.v.guarnizocabezas@bham.ac.uk

test -d ${BB_WORKDIR} && /bin/rm -rf ${BB_WORKDIR}

echo "Exiting."
exit 0
echo "Exited."
