#!/usr/bin/env python3

"""
**Image Recognition Script**

This script searches through a folder of images and identifies the objects within them.
It then saves the filename and a description of the objects found in a CSV file.

**Author:** Nasser Al-Hilal
**Date:** 16 Feb 2024
**Version:** 1.0

**Requirements:**
* Python 3
* OpenCV (for image processing)
* TensorFlow or PyTorch (for object detection)
* pandas (for CSV handling)

**Usage:**
```python
python image_recognition.py <image_folder> <output_file.csv>
"""

import argparse
import os

from pathlib import Path
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


def check_if_directory(directory_path):

    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        exit(-1)
    elif not os.path.isdir(directory_path):
        print(f"Not a directory: {directory_path}")
        exit(-1)
    else:
        return True


def find_image_files(directory):

    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = Path(file)
            if file_path.suffix.lower() in Image.registered_extensions():
                image_files.append(os.path.join(root, file))
    return image_files


def generate_caption(image_path, processor, model):
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=20)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption if caption else None


def main(directory):

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    ).to("cuda")

    image_files = find_image_files(directory)
    for i, image_path in enumerate(image_files):
        caption = generate_caption(image_path, processor, model)
        print(f"{i}: {image_path}: {caption}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory", type=str, help="Path to folder containing pictures to identify"
    )
    args = parser.parse_args()

    if args.directory and check_if_directory(args.directory):
        directory = args.directory
        main(directory)
    else:
        exit(-1)
