# Create an script to remove the checkpoints
# Get a list of games from the command line arguments
GAME_LIST=("$@")
# If no games are provided, default to a predefined list
if [ ${#GAME_LIST[@]} -eq 0 ]; then
    GAME_LIST=("AirRaid"
                "Alien" 
                "Amidar")
fi


# Print the games in a for loop
for GAME_NAME in "${GAME_LIST[@]}"; do
    echo "Synchronizing Game: $GAME_NAME"

    server_path="/rds/projects/g/giacobbm-bisimulation-rl/bisimulation-PER/logs/${GAME_NAME}/"
    local_path="logs/${GAME_NAME}/"

    # Check if the local path exists
    if [ ! -d "$local_path" ]; then
        echo "Local path $local_path does not exist. Creating it."
        mkdir -p "$local_path"
    fi

    # Synchronize the server path with the local path
    rsync -av --exclude='checkpoints' guarniov@bluebear.bham.ac.uk:"${server_path}" $local_path
done