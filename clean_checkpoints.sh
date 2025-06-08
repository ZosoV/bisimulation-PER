#!/bin/bash

if [ -n "$1" ] && [ -d "$1" ]; then
    # Change to the directory and get the latest numbered directory (excluding tmp folders)
    echo "Entering directory: $1"
    cd "$1" || exit 1
    # latest=$(ls -d [0-9]* 2>/dev/null | grep -v 'tmp' | sort -n | tail -1)

    # if [ -z "$latest" ]; then
    #     echo "No valid numbered directories found in $1 (excluding tmp folders)"
    #     exit 1
    # fi

    # echo "Keeping: $latest"
    # echo "Will delete:"
    # Check if the folder 98 exists and remove it send a message
    if [ -d "98" ]; then
        echo "98/"
        echo "Removing 98/ directory"
        rm -r 98/
    else
        echo "No 98/ directory found"
    fi  
    
else
    echo "Usage: $0 <directory>"
    echo "Directory must exist"
    exit 1
fi