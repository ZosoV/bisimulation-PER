#!/bin/bash
GAME_LIST=("$@")
# If no games are provided, default to a predefined list
if [ ${#GAME_LIST[@]} -eq 0 ]; then
    GAME_LIST=("AirRaid/"
                "Alien/" 
                "Amidar/")
fi

# Execute each game sequentially
for GAME_NAME in "${GAME_LIST[@]}"; do
    echo "Processing game: logs/${GAME_NAME}"

    # Find all checkpoints directories
    echo "The following checkpoints directories will be processed:"
    echo "------------------------------------------------------"
    find "logs/${GAME_NAME}" -type d -name "checkpoints" | while read -r checkpoint_dir; do
        echo "$checkpoint_dir"
    done
    echo "------------------------------------------------------"

    # Ask for confirmation
    read -p "Do you want to proceed with cleaning these directories? [y/N] " -n 1 -r
    echo    # move to a new line
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Operation cancelled."
        exit 0
    fi

    # Process each directory
    find "logs/${GAME_NAME}" -type d -name "checkpoints" | while read -r checkpoint_dir; do
        echo "Processing: $checkpoint_dir"
        bash clean_checkpoints.sh "$checkpoint_dir"
    done

    echo "Finished processing all checkpoints directories"

done
