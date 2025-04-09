#!/bin/bash
if [ -n "$1" ] && [ -d "$1" ]; then
    # Change to the directory and get the latest numbered directory
    cd "$1" && latest=$(ls -d [0-9]* 2>/dev/null | sort -n | tail -1)

    if [ -z "$latest" ]; then
        echo "No numbered directories found in $1"
        exit 1
    fi

    echo "Keeping: $latest"
    echo "Will delete:"
    # List files that will be deleted (dry run)
    ls | grep -v "^$latest$" | grep -v "sentinel_checkpoint_complete.$latest" | grep -v "ckpt.$latest"
    # Actually delete the files
    ls | grep -v "^$latest$" | grep -v "sentinel_checkpoint_complete.$latest" | grep -v "ckpt.$latest" | xargs rm -rf
else
    echo "Usage: $0 <directory>"
    echo "Directory must exist"
    exit 1
fi