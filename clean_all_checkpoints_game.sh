#!/bin/bash
if [ -z "$1" ] || [ ! -d "$1" ]; then
    echo "Usage: $0 <game_directory>"
    echo "Game directory must exist"
    exit 1
fi

# Find all checkpoints directories
echo "The following checkpoints directories will be processed:"
echo "------------------------------------------------------"
find "$1" -type d -name "checkpoints" | while read -r checkpoint_dir; do
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
find "$1" -type d -name "checkpoints" | while read -r checkpoint_dir; do
    echo "Processing: $checkpoint_dir"
    bash clean_checkpoints.sh "$checkpoint_dir"
done

echo "Finished processing all checkpoints directories"
