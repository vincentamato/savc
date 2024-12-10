import os

# Define the folder path
folder_path = "data/midis"

# Initialize a counter
midi_count = 0

# Walk through the folder and its subdirectories
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # Check if the file ends with .midi or .mid
        if file.lower().endswith(('.midi', '.mid')):
            midi_count += 1

print(f"Number of .midi or .mid files in '{folder_path}': {midi_count}")
