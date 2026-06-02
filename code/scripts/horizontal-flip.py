import os
import sys
from pathlib import Path
from PIL import Image


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(FILE_DIR))

def flip_png_images(source_dir: Path, dest_dir: Path) -> None:
    """
    Flips all PNG images in the source folder horizontally.
    """
    if not source_dir.is_dir():
        print(f"Error: '{source_dir}' is not a valid directory.")
        sys.exit(1)
 
    dest_dir.mkdir(parents=True, exist_ok=True)
 
    png_files = list(source_dir.glob("*.png")) + list(source_dir.glob("*.PNG"))
 
    if not png_files:
        print(f"No PNG files found in '{source_dir}'.")
        return
 
    print(f"{len(png_files)} PNG file(s) found. Processing...")
    
    for png_path in png_files:
        try:
            with Image.open(png_path) as img:
                flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
                output_path = dest_dir / png_path.name
                flipped.save(output_path)
        except Exception as e:
            print(f"Error with {png_path.name} : {e}")
 
    print("Completed.")
 
 
if __name__ == "__main__":
    SOURCE_DIR = os.path.join(ROOT_DIR, "data", "assets", "occluders")
    DEST_DIR = os.path.join(ROOT_DIR, "data", "assets", "occluders", "flipped")
 
    flip_png_images(
        source_dir=SOURCE_DIR, 
        dest_dir=DEST_DIR
    )