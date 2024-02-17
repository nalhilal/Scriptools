#!/usr/bin/env python3

# Image Recognition Script

# Author: Nasser Al-Hilal
# Date: 16 Feb 2024
# Version: 1.0

# Requirements:
# - Pytorch: install from https://pytorch.org/get-started/locally/
# - Pillow (image processing)
# - Tranformers (Hugging face model interaction)
# - argparse, csv, os, pathlib: system libs

# Usage:
# python clip-picture-search.py <image_folder>

# Import necessary libraries

import argparse
import csv
import os

from pathlib import Path
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


def get_csv_path(directory):

    full_path = os.path.abspath(directory)
    directory_name = os.path.basename(full_path)
    catalog_filename = f"catalog-{directory_name}.csv"
    catalog_path = os.path.join(directory, os.path.pardir, catalog_filename)
    return catalog_path


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


def write_output_to_csv(csv_table, csv_path):

    with open(csv_path, "+a", newline="", encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        for captions_row in csv_table:
            csvwriter.writerow(captions_row)
        csvfile.flush()
    print(f"Catalog data written to file: {csv_path}")


def main(directory):

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    ).to("cuda")

    image_files = find_image_files(directory)
    csv_table = []
    for i, image_path in enumerate(image_files):
        caption = generate_caption(image_path, processor, model)
        csv_row = [i, image_path, caption]
        csv_table.append(csv_row)
        print(f"{i}: {image_path}: {caption}")

    csv_file = get_csv_path(directory)
    write_output_to_csv(csv_table, csv_file)


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
