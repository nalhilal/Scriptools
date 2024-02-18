#!/usr/bin/env python3

# Image Recognition Script

# Author: Nasser Al-Hilal
# Date: 16 Feb 2024
# Version: 1.1

# Requirements:
# - Pytorch: (already in requiremetns.txt or can install from https://pytorch.org/get-started/locally/)
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
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM,
)
import torch


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


def generate_caption(image_path, processor, model, device, model_name):

    try:
        raw_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None

    try:
        model.to(device)
        if model_name == "blip":
            inputs = processor(images=raw_image, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_length=20)
            caption = processor.decode(outputs[0], skip_special_tokens=True)
        elif model_name == "git":
            inputs = processor(images=raw_image, return_tensors="pt").to(device)
            pixel_values = inputs.pixel_values
            generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"Error generating caption for {image_path}: {e}")
        return None

    return caption if caption else "No caption generated"


def write_output_to_csv(csv_table, csv_path):

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            csvwriter = csv.writer(csvfile)
            for captions_row in csv_table:
                csvwriter.writerow(captions_row)
    except Exception as e:
        print(f"Error writing to CSV {csv_path}: {e}")
    else:
        print(f"Catalog data written to file: {csv_path}")


def main(directory, model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if model_name == "blip":
        model_id = "Salesforce/blip-image-captioning-large"
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
    elif model_name == "git":
        model_id = "microsoft/git-large-textcaps"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    image_files = find_image_files(directory)

    csv_table = []
    for i, image_path in enumerate(image_files):
        caption = generate_caption(image_path, processor, model, device, model_name)
        csv_row = [i, image_path, caption]
        csv_table.append(csv_row)
        print(f"{i}: {image_path}: {caption}")

    if csv_table:
        csv_file = get_csv_path(directory)
        write_output_to_csv(csv_table, csv_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory", type=str, help="Path to folder containing pictures to identify"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["blip", "git"],
        default="blip",
        help="Select the model to use: 'blip' for Salesforce BLIP or 'git' for Microsoft GIT",
    )

    args = parser.parse_args()

    if args.directory and check_if_directory(args.directory):
        main(args.directory, args.model)
    else:
        exit(-1)
