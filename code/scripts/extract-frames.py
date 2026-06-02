import argparse
import cv2
import os
import numpy as np


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(FILE_DIR))

def apply_flip(image):
    return cv2.flip(image, 1)

def apply_rotation(image, angle):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)  # white background
    )

def apply_brightness_contrast(image, brightness=1.0, contrast=1.0):
    img = image.astype(np.float32)
    img = img * brightness
    mean = img.mean()
    img = (img - mean) * contrast + mean
    return np.clip(img, 0, 255).astype(np.uint8)

def apply_hue_saturation(image, hue_shift=0, sat_scale=1.0):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_gaussian_blur(image, ksize=3):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def build_augmentations(seed=42):
    """
    Build augmentation configurations. Each augmentation uses FIXED parameters
    per video to maintain temporal coherence across frames.
    """
    rng = np.random.default_rng(seed)
    
    rotation_angle = rng.uniform(-8, 8)
    brightness_factor = rng.uniform(0.85, 1.15)
    contrast_factor = rng.uniform(0.85, 1.15)
    hue_shift = rng.integers(-10, 10)
    sat_scale = rng.uniform(0.8, 1.2)
    
    augmentations = {
        "flipped": lambda img: apply_flip(img),
        "rotation": lambda img: apply_rotation(img, rotation_angle),
        "brightness": lambda img: apply_brightness_contrast(img, brightness=brightness_factor),
        "contrast": lambda img: apply_brightness_contrast(img, contrast=contrast_factor),
        "colorimetry": lambda img: apply_hue_saturation(img, hue_shift=hue_shift, sat_scale=sat_scale),
        "blur": lambda img: apply_gaussian_blur(img, ksize=3),
        "flipped_colorimetry": lambda img: apply_hue_saturation(
            apply_flip(img), hue_shift=hue_shift, sat_scale=sat_scale
        ),
    }
    
    return augmentations, {
        "rotation_angle": rotation_angle,
        "brightness_factor": brightness_factor,
        "contrast_factor": contrast_factor,
        "hue_shift": int(hue_shift),
        "sat_scale": sat_scale,
    }

def extract_dataset_frames(video_name, video_path, output_folder_path):
    os.makedirs(output_folder_path, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    
    if not vidcap.isOpened():
        print(f"Error: Unable to open the video file: {video_path}")
        return

    frame_count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if success:
            frame_filename = os.path.join(output_folder_path, f"{video_name}_{frame_count:05d}.png")
            cv2.imwrite(frame_filename, image)
            frame_count += 1
    
    vidcap.release()
    print(f"Extraction completed. {frame_count} images extracted to {output_folder_path}")

def extract_frames(video_path, output_folder_path):
    os.makedirs(output_folder_path, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    
    if not vidcap.isOpened():
        print(f"Error: Unable to open the video file: {video_path}")
        return

    frame_count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if success:
            frame_filename = os.path.join(output_folder_path, f"{frame_count:05d}.png")
            cv2.imwrite(frame_filename, image)
            frame_count += 1
    
    vidcap.release()
    print(f"Extraction completed. {frame_count} images extracted to {output_folder_path}")

def extract_frames_with_augmentations(video_path, video_name, base_output_dir, augmentations):
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print(f"Error: Unable to open the video file: {video_path}")
        return

    # Prepare output folders
    output_folders = {
        "original": os.path.join(base_output_dir, video_name, "original", "frames")
    }
    for suffix in augmentations.keys():
        output_folders[suffix] = os.path.join(
            base_output_dir, f"{video_name}", f"{suffix}", "frames"
        )
    
    for folder in output_folders.values():
        os.makedirs(folder, exist_ok=True)

    frame_count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if not success:
            break

        # Original frame
        cv2.imwrite(
            os.path.join(output_folders["original"], f"{frame_count:05d}.png"),
            image
        )

        # Augmented frames
        for suffix, transform in augmentations.items():
            try:
                aug_image = transform(image)
                cv2.imwrite(
                    os.path.join(output_folders[suffix], f"{frame_count:05d}.png"),
                    aug_image
                )
            except Exception as e:
                print(f"  [Warning] Failed augmentation '{suffix}' on frame {frame_count}: {e}")

        frame_count += 1

    vidcap.release()

    print(f"\nExtraction completed for '{video_name}'. {frame_count} frames per version.")
    for suffix in augmentations.keys():
        print(f"  {suffix:16s} -> {output_folders[suffix]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ExtractFrames',
        description='Extract all frames from a video and optionally generate augmented versions'
    )
    parser.add_argument('-video_filename', type=str, required=True,
                        help='Name of the video file to extract frames from')
    parser.add_argument('--augment', action='store_true',
                        help='Enable augmentations (generates additional augmented folders)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for augmentation parameters')
    args = parser.parse_args()

    VIDEO_FILENAME = args.video_filename
    VIDEO_NAME = os.path.splitext(VIDEO_FILENAME)[0]
    VIDEO_PATH = os.path.join(ROOT_DIR, "data", "inputs", VIDEO_FILENAME)
    BASE_OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "outputs")

    if args.augment:
        augmentations, params = build_augmentations(seed=args.seed)
        print(f"Augmentation parameters for '{VIDEO_NAME}':")
        for k, v in params.items():
            print(f"  {k}: {v}")
        print()

        extract_frames_with_augmentations(
            video_path=VIDEO_PATH,
            video_name=VIDEO_NAME,
            base_output_dir=BASE_OUTPUT_DIR,
            augmentations=augmentations
        )
    else:
        output_folder_path = os.path.join(BASE_OUTPUT_DIR, VIDEO_NAME, "frames")
        extract_frames(video_path=VIDEO_PATH, output_folder_path=output_folder_path)