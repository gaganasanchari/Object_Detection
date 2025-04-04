import os
import shutil
import random

# Paths
image_dir = "./images/"
label_dir = "./labels/"
train_img_dir = "./images/train/"
val_img_dir = "./images/val/"
train_lbl_dir = "./labels/train/"
val_lbl_dir = "./labels/val/"

# Ensure directories exist
for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
    os.makedirs(d, exist_ok=True)

# Get all images
all_images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
random.shuffle(all_images)

# Define train/val split (80-20)
split_idx = int(0.8 * len(all_images))
train_images, val_images = all_images[:split_idx], all_images[split_idx:]

# Function to find the correct label filename
def get_label_filename(image_name):
    base_name = os.path.splitext(image_name)[0].replace("Imges_", "")  # Remove .jpg/.png and 'Imges_'
    return f"labels_{base_name}.txt"  # Custom label format

# Move images and corresponding labels
def move_files(image_list, dest_img_dir, dest_lbl_dir):
    for img in image_list:
        shutil.move(os.path.join(image_dir, img), os.path.join(dest_img_dir, img))
        lbl = get_label_filename(img)
        if os.path.exists(os.path.join(label_dir, lbl)):
            shutil.move(os.path.join(label_dir, lbl), os.path.join(dest_lbl_dir, lbl))
        else:
            print(f"⚠️ Warning: Label missing for {img} (Expected: {lbl})")

# Move training and validation sets
move_files(train_images, train_img_dir, train_lbl_dir)
move_files(val_images, val_img_dir, val_lbl_dir)

print(f"✅ Split complete! Train: {len(train_images)}, Val: {len(val_images)}")

