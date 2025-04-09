# I have only 1TB of disk space, and I want to make sure that I have enough space to run the code

game_consumption = 110 # in GB including all seeds
num_games = 9

total_consumption = game_consumption * num_games
total_consumption = total_consumption / 1024 # in TB
print(f"Total consumption: {total_consumption} TB")
if total_consumption > 1:
    print("Not enough disk space")
else:
    print("Enough disk space")

