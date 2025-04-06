# Create an script to remove the checkpoints
GAME_NAME=${1:-Alien}  # Default to Alien if no game name is specified


server_path="/rds/projects/g/giacobbm-bisimulation-rl/bisimulation-PER/logs/${GAME_NAME}/"
local_path="logs/${GAME_NAME}/"

# Check if the local path exists
if [ ! -d "$local_path" ]; then
    echo "Local path $local_path does not exist. Creating it."
    mkdir -p "$local_path"
fi

# Synchronize the server path with the local path
rsync -av --exclude='checkpoints' guarniov@bluebear.bham.ac.uk:"${server_path}" $local_path