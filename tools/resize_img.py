#!/usr/bin/env python3
import os
import sys
import argparse
from PIL import Image


def resize_images(directory, max_width, max_height):
    max_pixels = max_width * max_height

    for filename in os.listdir(directory):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            file_path = os.path.join(directory, filename)

            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    current_pixels = width * height

                    if current_pixels > max_pixels:
                        # Calculate new dimensions while maintaining aspect ratio
                        ratio = (max_pixels / current_pixels) ** 0.5
                        new_width = int(width * ratio)
                        new_height = int(height * ratio)

                        print(
                            f"Resizing {filename}: {width}x{height} ({current_pixels} pixels) -> {new_width}x{new_height} ({new_width * new_height} pixels)"
                        )

                        # Resize and save with original format
                        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                        resized_img.save(file_path, quality=95)
                    else:
                        print(
                            f"Skipping {filename}: {width}x{height} ({current_pixels} pixels) <= {max_pixels} pixels"
                        )
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Resize images in a directory if they exceed a pixel count threshold"
    )
    parser.add_argument(
        "--resolution", help="Maximum resolution in format HEIGHTxWIDTH (e.g. 1080x1920)"
    )
    parser.add_argument("--imgdir", help="Directory containing images to process")

    args = parser.parse_args()

    if not args.resolution or "x" not in args.resolution:
        print("Error: --resolution must be specified in format HEIGHTxWIDTH (e.g. 1080x1920)")
        sys.exit(1)

    try:
        height, width = map(int, args.resolution.lower().split("x"))
        height -= 16
        width -= 16
    except ValueError:
        print("Error: Invalid resolution format. Use HEIGHTxWIDTH (e.g. 1080x1920)")
        sys.exit(1)

    if not os.path.isdir(args.imgdir):
        print(f"Error: {args.imgdir} is not a valid directory")
        sys.exit(1)

    resize_images(args.imgdir, width, height)


if __name__ == "__main__":
    main()
