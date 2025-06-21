import kagglehub

# Download latest version
path = kagglehub.dataset_download("vekosek/cats-from-memes")

print("Path to dataset files:", path)