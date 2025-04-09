#!/bin/bash
if [ -z "$1" ] || [ ! -d "$1" ]; then
    echo "Usage: $0 <game_directory>"
    echo "Game directory must exist"
    exit 1
fi

# Find all checkpoints directories
find "$1" -type d -name "checkpoints" | while read -r checkpoint_dir; do
    echo "Processing: $checkpoint_dir"
    # Call your original script for each checkpoints directory
    bash clean_checkpoints.sh "$checkpoint_dir"
done

echo "Finished processing all checkpoints directories"
