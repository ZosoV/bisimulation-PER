#!/bin/bash
#SBATCH --job-name=bisimulation-rl-DQN
#SBATCH --array=0-2
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=o.v.guarnizocabezas@bham.ac.uk
#SBATCH --qos=bbgpu
#SBATCH --cpus-per-task=14
#SBATCH --account=giacobbm-bisimulation-rl
#SBATCH --gres=gpu:a100:1
#SBATCH --output="outputs/slurm-files/slurm-DQN-%A_%a.out"

module purge; module load bluebear
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0
module load cuDNN/8.9.2.26-CUDA-12.1.1


# module load tqdm/4.66.1-GCCcore-12.3.0

# module purge; module load bluebear
# module load bear-apps/2022a
# module load Python/3.10.4-GCCcore-11.3.0
# module load tqdm/4.64.0-GCCcore-11.3.0
# module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
# module load jax/0.4.4-foss-2022a-CUDA-11.7.0
# module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
# module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
# module load bear-apps/2022a
# module load wandb/0.13.6-GCC-11.3.0

GAME_NAME=${1:-Alien}  # Default to Alien if no game name is specified
AGENT_NAME=${AGENT_NAME:-metric_dqn_bper}  # Default to metric_dqn_bper if no agent name is specified
BPER_SCHEME=${BPER_SCHEME:-"scaling"}  # Default to softmax if no BPER scheme is specified

# CUSTOM_THREADS=18
echo "Game name: $GAME_NAME"
echo "Agent name: $AGENT_NAME"

# Temporary scratch space for I/O efficiency
BB_WORKDIR=$(mktemp -d /scratch/${USER}_${SLURM_JOBID}.XXXXXX)
# BB_WORKDIR=$(mktemp -d /rds/projects/g/giacobbm-bisimulation-rl/${USER}_${SLURM_JOBID}.XXXXXX)
export TMPDIR=${BB_WORKDIR}
# export EXP_BUFF=${BB_WORKDIR}

# Check if an argument is provided
# if [ -z "$1" ]; then
#     echo "Error: No W&B API key provided."
#     exit 1
# fi

# Set W&B API key from argument and dir
# export WANDB_API_KEY=$1
# export WANDB_DIR=${BB_WORKDIR}/wandb
# mkdir -p $WANDB_DIR

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

seeds=(118398 919409 711872 442081 189061)

# Select the seed based on the SLURM array task ID
SEED=${seeds[$SLURM_ARRAY_TASK_ID]}

# Record the start time
start_time=$(date +%s)
echo "Starting task with seed $SEED at $(date)"

# Define a function to be executed on exit
function notify_job_completion {
    exit_status=$?
    end_time=$(date +%s)
    runtime=$((end_time - start_time))
    days=$((runtime / 86400))
    hours=$(( (runtime % 86400) / 3600 ))
    minutes=$(( (runtime % 3600) / 60 ))

    slurm_output_file="outputs/slurm-files/slurm-DQN-${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}.out"

    subject="SLURM Job: ${AGENT_NAME} on ${GAME_NAME} (Seed: ${SEED})"
    if [ $exit_status -ne 0 ]; then
        subject="${subject} [FAILED]"
    else
        subject="${subject} [COMPLETED]"
    fi

    {
        echo "Task finished at $(date)"
        echo "Exit status: $exit_status"
        echo "Seed: $SEED"
        echo "Game: $GAME_NAME"
        echo "Agent: $AGENT_NAME"
        echo "Total runtime: ${days} days, ${hours} hours, ${minutes} minutes"
        echo ""
        # echo "SLURM Output (${slurm_output_file}):"
        # cat "$slurm_output_file"
    } | mail -s "$subject" o.v.guarnizocabezas@bham.ac.uk
}

# Trap both EXIT and ERR signals
trap notify_job_completion EXIT

# Print current OMP_NUM_THREADS and MKL_NUM_THREADS
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
echo "NUM_INTER_THREADS=$NUM_INTER_THREADS"
echo "NUM_INTRA_THREADS=$NUM_INTRA_THREADS"
echo "XLA_FLAGS=$XLA_FLAGS"

# # Set the number of threads for MKL and OMP
# export OMP_NUM_THREADS=$CUSTOM_THREADS
# export MKL_NUM_THREADS=$CUSTOM_THREADS

# Execute based on the selected variant
if [ "$AGENT_NAME" == "metric_dqn_bper" ]; then

    python -m train \
        --base_dir=logs/ \
        --gin_files=dqn.gin \
        --game_name=${GAME_NAME} \
        --agent_name=${AGENT_NAME} \
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
    
else
    echo "Unknown variant: $AGENT_NAME"
    echo "Task with seed: $SEED, game: $GAME_NAME and agent: $AGENT_NAME has failed at $(date)" | mail -s "SLURM Job Notification: Task Failed" o.v.guarnizocabezas@bham.ac.uk
    exit 1
fi

echo "Completed task with seed $SEED at $(date)"

# Removing extra checkpoints
# echo "Removing extra checkpoints"
# CHECKPOINTS_DIR="logs/${GAME_NAME}/${AGENT_NAME}/${SEED}/checkpoints"

# if [ -n "${CHECKPOINTS_DIR}" ] && [ -d "${CHECKPOINTS_DIR}" ]; then
#     # Change to the directory and get the latest numbered directory
#     cd "${CHECKPOINTS_DIR}" && latest=$(ls -d [0-9]* 2>/dev/null | sort -n | tail -1)

#     if [ -z "$latest" ]; then
#         echo "No numbered directories found in ${CHECKPOINTS_DIR}"
#         exit 1
#     fi

#     echo "Keeping: $latest"
#     echo "Will delete:"
#     # List files that will be deleted (dry run)
#     ls | grep -v "^$latest$" | grep -v "sentinel_checkpoint_complete.$latest" | grep -v "ckpt.$latest"
#     # Actually delete the files
#     ls | grep -v "^$latest$" | grep -v "sentinel_checkpoint_complete.$latest" | grep -v "ckpt.$latest" | xargs rm -rf
# else
#     echo "Usage: $0 <directory>"
#     echo "Directory must exist"
#     exit 1
# fi

# Cleanup
# sleep 300  # 5-minute buffer
# test -d ${BB_WORKDIR}/wandb/ && /bin/cp -r ${BB_WORKDIR}/wandb/ ./outputs/wandb/
test -d ${BB_WORKDIR} && /bin/rm -rf ${BB_WORKDIR}

echo "Exiting."
exit 0
echo "Exited."
