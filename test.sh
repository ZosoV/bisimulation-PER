seeds=(118398 919409 711872) # 442081 189061)
GAME_NAME="PongNoFrameskip-v4"
set -x  # Enable debug mode

# Execute each game sequentially
AGENT_NAME="metric_dqn_bper"
BPER_SCHEME="softmax"
for SEED in "${seeds[@]}"; do
    echo "Game name: $GAME_NAME"
    echo "Agent name: $AGENT_NAME"

    # Record the start time for the current game
    game_start_time=$(date +%s)
    echo "Starting game $GAME_NAME at $(date)"

    # Execute based on the selected variant
    # python -m train \
    #     --base_dir=logs/ \
    #     --gin_files=dqn.gin \
    #     --game_name=${GAME_NAME} \
    #     --agent_name="${AGENT_NAME}_${BPER_SCHEME}" \
    #     --seed=${SEED} \
    #     --gin_bindings="MetricDQNBPERAgent.method_scheme='${BPER_SCHEME}'"

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
        echo "BPER Scheme: $BPER_SCHEME"
        echo "Seed: $SEED"
        echo "Runtime: ${days}d ${hours}h ${minutes}m"
        echo ""
        echo "SLURM Job ID: $SLURM_JOBID"
    } | mailx -s "Slurm Array Job: [Game Completed] ${AGENT_NAME} on ${GAME_NAME} (Seed: ${SEED})" o.v.guarnizocabezas@bham.ac.uk

    echo "Completed game $GAME_NAME at $(date)"
done


AGENT_NAME="metric_dqn_bper"
BPER_SCHEME="scaling"
for SEED in "${seeds[@]}"; do
    echo "Game name: $GAME_NAME"
    echo "Agent name: $AGENT_NAME"

    # Record the start time for the current game
    game_start_time=$(date +%s)
    echo "Starting game $GAME_NAME at $(date)"

    # Execute based on the selected variant
    # python -m train \
    #     --base_dir=logs/ \
    #     --gin_files=dqn.gin \
    #     --game_name=${GAME_NAME} \
    #     --agent_name="${AGENT_NAME}_${BPER_SCHEME}" \
    #     --seed=${SEED} \
    #     --gin_bindings="MetricDQNBPERAgent.method_scheme='${BPER_SCHEME}'"

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
        echo "BPER Scheme: $BPER_SCHEME"
        echo "Seed: $SEED"
        echo "Runtime: ${days}d ${hours}h ${minutes}m"
        echo ""
        echo "SLURM Job ID: $SLURM_JOBID"
    } | mailx -s "Slurm Array Job: [Game Completed] ${AGENT_NAME} on ${GAME_NAME} (Seed: ${SEED})" o.v.guarnizocabezas@bham.ac.uk

    echo "Completed game $GAME_NAME at $(date)"
done

AGENT_NAME="metric_dqn_per"
for SEED in "${seeds[@]}"; do
    echo "Game name: $GAME_NAME"
    echo "Agent name: $AGENT_NAME"

    # Record the start time for the current game
    game_start_time=$(date +%s)
    echo "Starting game $GAME_NAME at $(date)"

    # Execute based on the selected variant
    # python -m train \
    #     --base_dir=logs/ \
    #     --gin_files=dqn.gin \
    #     --game_name=${GAME_NAME} \
    #     --agent_name=${AGENT_NAME} \
    #     --seed=${SEED} \
    #     --gin_bindings="MetricDQNBPERAgent.bper_weight=0"

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