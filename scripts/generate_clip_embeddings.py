import os
import sys
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

def generate_embeddings(image_dir, output_path="embeddings.npy"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    # Gather all image paths
    image_files = []
    for root, dirs, files in os.walk(image_dir):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fpath = os.path.join(root, fname)
                image_files.append(fpath)

    if len(image_files) == 0:
        print("No valid image files found in the directory.")
        sys.exit(1)

    embeddings = []
    image_paths = []

    # Process images with a progress bar
    for fpath in tqdm(image_files, desc="Processing images", unit="image"):
        try:
            image = Image.open(fpath).convert("RGB")
        except Exception as e:
            print(f"Error loading image {fpath}: {e}")
            continue

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        embeddings.append(image_features.squeeze().cpu().numpy())
        rel_path = os.path.relpath(fpath, image_dir)
        image_paths.append(rel_path)

    if len(embeddings) == 0:
        print("No embeddings generated. Check if the images are valid and supported.")
        sys.exit(1)

    embeddings = np.stack(embeddings, axis=0)
    np.save(output_path, embeddings)

    with open(output_path.replace(".npy", "_files.txt"), "w") as f:
        for p in image_paths:
            f.write(p + "\n")

    print(f"Saved embeddings to {output_path} and image list to {output_path.replace('.npy', '_files.txt')}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clip_hf_embeddings.py <image_directory> [output_embeddings.npy]")
        sys.exit(1)

    img_dir = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "embeddings.npy"
    generate_embeddings(img_dir, out_path)
