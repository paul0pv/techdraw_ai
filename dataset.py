import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

# Define input and output directories
input_dirs = {"urban_real": "urban_real", "architectural_style": "architectural_style"}

output_base = "cycleGAN_dataset"
output_dirs = {
    "trainA": os.path.join(output_base, "trainA"),
    "trainB": os.path.join(output_base, "trainB"),
    "testA": os.path.join(output_base, "testA"),
    "testB": os.path.join(output_base, "testB"),
}

# Create output directories
for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)


# Function to preprocess and save images
def preprocess_images(input_folder, train_folder, test_folder):
    image_files = [
        f
        for f in os.listdir(input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    train_files, test_files = train_test_split(
        image_files, test_size=0.2, random_state=42
    )

    for file_list, target_folder in [
        (train_files, train_folder),
        (test_files, test_folder),
    ]:
        for file_name in file_list:
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(target_folder, file_name)

            try:
                with Image.open(input_path) as img:
                    img = img.convert("RGB")
                    img = img.resize((256, 256), Image.BICUBIC)
                    img_array = np.asarray(img).astype(np.float32)
                    img_array = (img_array / 127.5) - 1.0  # Normalize to [-1, 1]
                    img = Image.fromarray(((img_array + 1.0) * 127.5).astype(np.uint8))
                    img.save(output_path)
            except Exception as e:
                print(f"Error processing {input_path}: {e}")


# Preprocess both domains
preprocess_images(input_dirs["urban_real"], output_dirs["trainA"], output_dirs["testA"])
preprocess_images(
    input_dirs["architectural_style"], output_dirs["trainB"], output_dirs["testB"]
)

print("Preprocessing complete. Images saved in 'cycleGAN_dataset' directory.")
