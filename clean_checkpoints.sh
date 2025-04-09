#!/bin/bash

if [ -n "$1" ] && [ -d "$1" ]; then
    cd "$1" || exit 1
    
    # Get latest numbered directory (excluding tmp folders)
    latest=$(ls -d [0-9]* 2>/dev/null | grep -v 'tmp' | sort -n | tail -1)

    if [ -z "$latest" ]; then
        echo "No valid numbered directories found in $1 (excluding tmp folders)"
        exit 1
    fi

    echo "Directory: $PWD"
    echo "Keeping: $latest"
    echo "Will delete:"
    
    # Dry run - show what would be deleted
    to_delete=$(ls | grep -v 'tmp' | grep -v "^$latest$" | 
               grep -v "sentinel_checkpoint_complete.$latest" | 
               grep -v "ckpt.$latest")
    
    if [ -z "$to_delete" ]; then
        echo "Nothing to delete in $PWD"
        exit 0
    fi
    
    echo "$to_delete"
    echo "----------------------------------------"
    
    # Ask for confirmation
    read -p "Confirm deletion? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping $PWD"
        exit 0
    fi
    
    # Actual deletion
    echo "$to_delete" | xargs rm -rf
    echo "Deleted files in $PWD"
else
    echo "Usage: $0 <directory>"
    echo "Directory must exist"
    exit 1
fi