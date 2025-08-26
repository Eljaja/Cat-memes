import kagglehub
import os
import shutil

# Download to default location first
default_path = kagglehub.dataset_download("vekosek/cats-from-memes")

# Move to current directory
current_dir = os.getcwd()
target_dir = os.path.join(current_dir, "cats_from_memes")

# Remove target directory if it exists
if os.path.exists(target_dir):
    shutil.rmtree(target_dir)

# Move the downloaded dataset
shutil.move(default_path, target_dir)

print("Dataset downloaded to:", target_dir)