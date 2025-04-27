#!/bin/bash

# Get a list of games from the command line arguments
GAME_LIST=("$@")
# If no games are provided, default to a predefined list
if [ ${#GAME_LIST[@]} -eq 0 ]; then
    GAME_LIST=("AirRaid"
                "Alien" 
                "Amidar")
fi

# Print the list of games
echo "Game list: ${GAME_LIST[@]}"
# Print the number of games
echo "Number of games: ${#GAME_LIST[@]}"

# Print the games in a for loop
for GAME in "${GAME_LIST[@]}"; do
    echo "Game: $GAME"
done