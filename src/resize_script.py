import os
from PIL import Image
from tqdm import tqdm

def resize_images_inplace(img_dir, target_size=(224, 224)):
    """Resize all images in directory to target_size, overwriting originals"""

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Found {len(img_files)} images to resize")

    for filename in tqdm(img_files, desc="Resizing images"):
        img_path = os.path.join(img_dir, filename)

        try:
            with Image.open(img_path) as img:
                # Convert to RGB if needed (handles any format issues)
                img = img.convert("RGB")

                # Resize using high-quality resampling
                img_resized = img.resize(target_size, Image.Resampling.LANCZOS)

                # Save back to same location (overwrites original)
                img_resized.save(img_path, 'PNG', optimize=True)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    img_dir = "res/data/hateful_memes_data/img"
    resize_images_inplace(img_dir)
    print("Resizing complete!")