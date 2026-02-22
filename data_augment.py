#use roboflow.com to augment data and label it for yolo model training

import cv2
import os
import albumentations as A

# Input & output folders
input_folder = "photos"
output_folder = "photos_augmented"

os.makedirs(output_folder, exist_ok=True)

# Augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3),
    A.Blur(p=0.2),
    A.RandomScale(scale_limit=0.2, p=0.5)
])

count = 0

for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)
    image = cv2.imread(img_path)

    for i in range(10):  # 5 augmentations per image → 30 × 5 = 150
        augmented = transform(image=image)["image"]

        new_name = f"{img_name.split('.')[0]}_aug_{i}.jpg"
        cv2.imwrite(os.path.join(output_folder, new_name), augmented)
        count += 1

print(f"Generated {count} images")