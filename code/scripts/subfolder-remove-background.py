import argparse
import glob
import os
import cv2
import numpy as np
from tqdm import tqdm

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(FILE_DIR))

def main(masks_parent_folder_path, image_folder_path, output_folder_path):
    # Create output directory if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)

    images_list = sorted(
        [
            image
            for image in glob.glob(os.path.join(image_folder_path, "*.png"))
        ]
    )

    # Process each image
    for image_path in tqdm(images_list):
        base_name = os.path.basename(image_path)[:-4]
        human_mask_path = os.path.join(masks_parent_folder_path, f"{base_name}.png")
        
        if not os.path.exists(human_mask_path):
            continue

        image = cv2.imread(image_path)
        human_mask = cv2.imread(human_mask_path, cv2.IMREAD_GRAYSCALE)

        # Alignment of dimensions
        if human_mask.shape != image.shape[:2]:
            human_mask = cv2.resize(human_mask, (image.shape[1], image.shape[0]))

        # Pixel extraction
        # White image and copy only the pixels from the mask
        white_bg = np.ones_like(image) * 255 # White background
        mask_3ch = cv2.cvtColor(human_mask, cv2.COLOR_GRAY2BGR) if len(human_mask.shape) == 2 else human_mask
        result = np.where(mask_3ch > 0, image, white_bg)

        output_path = os.path.join(output_folder_path, f"{base_name}.png")
        cv2.imwrite(output_path, result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='RemoveBackground', description='Remove background from images using segmentation masks')
    parser.add_argument('-video_filename', type=str, help='Name of the video file to run inference on')
    args = parser.parse_args()

    VIDEO_FILENAME = args.video_filename
    VIDEO_NAME = VIDEO_FILENAME[:-4] # Remove file extension
    PARENT_FOLDER = os.path.join(ROOT_DIR, "data", "outputs", VIDEO_NAME)
    
    for folder in os.listdir(PARENT_FOLDER):
        IMAGE_FOLDER_PATH = os.path.join(PARENT_FOLDER, folder, "frames")
        MASKS_PARENT_FOLDER_PATH = os.path.join(PARENT_FOLDER, folder, "sam3_filtered")
        OUTPUT_PATH = os.path.join(PARENT_FOLDER, folder, "frames_without_background")

        main(
            masks_parent_folder_path=MASKS_PARENT_FOLDER_PATH,
            image_folder_path=IMAGE_FOLDER_PATH,
            output_folder_path=OUTPUT_PATH
        )
