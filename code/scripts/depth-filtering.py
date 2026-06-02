import argparse
import glob
import os
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from scipy import ndimage

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(FILE_DIR))


def load_moge_depth_metric(path):
    """
    Loads a 16-bit MoGe PNG and converts it to metres.
    """
    with Image.open(path) as img:
        near = float(img.info.get('near', 0.1))
        far = float(img.info.get('far', 100.0))
        depth_encoded = np.array(img).astype(np.float32)

    mask_finite = (depth_encoded > 0) & (depth_encoded < 65535)

    depth_metric = np.full_like(depth_encoded, np.nan)
    depth_metric[mask_finite] = near * np.power(
        far / near,
        (depth_encoded[mask_finite] - 1) / 65533
    )
    depth_metric[depth_encoded == 65535] = 1000.0
    return depth_metric

def keep_closest_person(mask_path, depth_path, output_path, min_pixels=500):
    """
    Filters a SAM3 person mask to retain only the athlete in the foreground,
    discarding other people
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return

    depth = load_moge_depth_metric(depth_path)
    if mask.shape[:2] != depth.shape[:2]:
        depth = cv2.resize(depth, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, os.path.basename(mask_path))

    # Label connected components
    labeled, n = ndimage.label(mask > 0)
    if n == 0:
        cv2.imwrite(out_file, np.zeros_like(mask))
        return

    sizes = ndimage.sum(mask > 0, labeled, range(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= min_pixels
    valid_mask = keep[labeled]

    if not valid_mask.any():
        cv2.imwrite(out_file, np.zeros_like(mask))
        return

    # Median of valid mask pixels
    valid_depths = depth[valid_mask & np.isfinite(depth)]
    if valid_depths.size == 0:
        cv2.imwrite(out_file, np.zeros_like(mask))
        return

    ref_depth = np.median(valid_depths)

    # Adaptive depth window
    window = max(0.4, 0.3 * ref_depth)

    # Per-pixel depth filter
    foreground = (
        valid_mask
        & np.isfinite(depth)
        & (depth >= ref_depth - window)
        & (depth <= ref_depth + window)
    ).astype(np.uint8) * 255

    # Morphological closing to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

    # Keep only the largest connected component
    labeled, n = ndimage.label(foreground > 0)
    if n == 0:
        cv2.imwrite(out_file, np.zeros_like(mask))
        return

    final_mask = np.zeros_like(foreground)
    for blob_id in range(1, n + 1):
        blob = labeled == blob_id
        blob_size = blob.sum()
        if blob_size < min_pixels:
            continue  # ignore small blobs

        blob_depths = depth[blob & np.isfinite(depth)]
        if blob_depths.size == 0:
            continue
        blob_median_depth = np.median(blob_depths)
        if abs(blob_median_depth - ref_depth) <= window:
            final_mask[blob] = 255

    cv2.imwrite(out_file, final_mask)
    
def main(image_folder_path, masks_folder_path, depth_folder_path, output_folder_path):
    os.makedirs(output_folder_path, exist_ok=True)

    images_list = sorted(glob.glob(os.path.join(image_folder_path, "*.png")))

    for image_path in tqdm(images_list):
        base_name = os.path.basename(image_path)[:-4]
        human_mask_path = os.path.join(masks_folder_path, f"{base_name}.png")
        depth_path = os.path.join(depth_folder_path, f"{base_name}.png")

        if not (os.path.exists(human_mask_path) and os.path.exists(depth_path)):
            continue

        keep_closest_person(human_mask_path, depth_path, output_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='DepthThresholdFiltering', description='Run depth threshold filtering on a folder of masks and depth maps')
    parser.add_argument('-video_filename', type=str, help='Name of the video file to run inference on')
    args = parser.parse_args()

    VIDEO_FILENAME = args.video_filename
    VIDEO_NAME = VIDEO_FILENAME[:-4] # Remove file extension
    IMAGE_FOLDER_PATH = os.path.join(ROOT_DIR, "data", "outputs", VIDEO_NAME, "frames")
    MASKS_FOLDER_PATH = os.path.join(ROOT_DIR, "data", "outputs", VIDEO_NAME, "sam3")
    DEPTH_FOLDER_PATH = os.path.join(ROOT_DIR, "data", "outputs", VIDEO_NAME, "moge")
    OUTPUT_PATH = os.path.join(ROOT_DIR, "data", "outputs", VIDEO_NAME, "sam3_filtered")

    main(
        image_folder_path=IMAGE_FOLDER_PATH,
        masks_folder_path=MASKS_FOLDER_PATH,
        depth_folder_path=DEPTH_FOLDER_PATH,
        output_folder_path=OUTPUT_PATH
    )
