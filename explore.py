import os
from PIL import Image
import matplotlib.pyplot as plt

# Path to your dataset
dataset_path = r"C:\aquavision\data\Augmented_shrimp"

# Step 1 — Count images per class
print("=== IMAGE COUNT PER CLASS ===")
class_counts = {}
for class_name in os.listdir(dataset_path):
    class_folder = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_folder):
        count = len(os.listdir(class_folder))
        class_counts[class_name] = count
        print(f"{class_name}: {count} images")

print(f"\nTotal classes: {len(class_counts)}")
print(f"Total images: {sum(class_counts.values())}")

# Step 2 — Check image dimensions
print("\n=== SAMPLE IMAGE DIMENSIONS ===")
for class_name in class_counts:
    class_folder = os.path.join(dataset_path, class_name)
    first_image = os.listdir(class_folder)[0]
    img_path = os.path.join(class_folder, first_image)
    img = Image.open(img_path)
    print(f"{class_name}: {img.size} — mode: {img.mode}")

# Step 3 — Display sample images
print("\n=== DISPLAYING SAMPLE IMAGES ===")
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("One Sample From Each Class", fontsize=14)

for idx, class_name in enumerate(class_counts):
    class_folder = os.path.join(dataset_path, class_name)
    first_image = os.listdir(class_folder)[0]
    img_path = os.path.join(class_folder, first_image)
    img = Image.open(img_path)
    axes[idx].imshow(img)
    axes[idx].set_title(class_name)
    axes[idx].axis("off")

plt.tight_layout()
plt.show()