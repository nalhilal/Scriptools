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


def check_if_directory(directory_path):
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        exit(-1)
    elif not os.path.isdir(directory_path):
        print(f"Not a directory: {directory_path}")
        exit(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory", type=str, help="Path to folder containing pictures to identify"
    )
    args = parser.parse_args()

    if args.directory and check_if_directory(args.directory):
        directory = args.directory
    else:
        exit(-1)