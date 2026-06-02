import argparse
import cv2
import numpy as np
import os
import random
from glob import glob
from tqdm import tqdm


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(FILE_DIR))

def overlay_transparent(background, overlay, x, y):
    """
    Overlays a PNG image onto a background at position (x, y).
    """
    bg_h, bg_w, _ = background.shape
    h, w, _ = overlay.shape

    # Check that the object does not extend beyond the edges of the background
    if x >= bg_w or y >= bg_h:
        return background

    # Clamp x and y so they are not negative
    if x < 0:
        overlay = overlay[:, -x:]
        w = overlay.shape[1]
        x = 0
    if y < 0:
        overlay = overlay[-y:, :]
        h = overlay.shape[0]
        y = 0

    # Adjust the size if the object extends beyond the right or bottom edge
    if x + w > bg_w: w = bg_w - x
    if y + h > bg_h: h = bg_h - y

    overlay = overlay[0:h, 0:w]

    if h <= 0 or w <= 0:
        return background

    # Separate the channels
    alpha = overlay[:, :, 3] / 255.0 
    img_rgb = overlay[:, :, :3]

    for c in range(3):
        background[y:y+h, x:x+w, c] = (alpha * img_rgb[:, :, c] + (1.0 - alpha) * background[y:y+h, x:x+w, c])
    return background

def get_occluder_type(occluder_path):
    name = os.path.basename(occluder_path).lower()
    if "_plate" in name:
        return "plate"
    elif "_rack" in name:
        return "rack"
    return "unknown"

def get_anchor_position(mask, occluder_h, occluder_w, margin=50):
    """
    Find an anchor position (x, y) near the person using the mask.
    """
    y_indices, x_indices = np.where(mask > 0)

    if len(x_indices) == 0:
        return (random.randint(0, max(1, mask.shape[1] - occluder_w)),
                random.randint(0, max(1, mask.shape[0] - occluder_h)))

    # Choose a random pixel on the person as an anchor point
    random_idx = random.randint(0, len(x_indices) - 1)
    target_x = x_indices[random_idx]
    target_y = y_indices[random_idx]

    # Add a random margin
    shift_x = random.randint(-margin, margin)
    shift_y = random.randint(-margin, margin)

    final_x = target_x + shift_x - (occluder_w // 2)
    final_y = target_y + shift_y - (occluder_h // 2)

    final_x = max(0, min(final_x, mask.shape[1] - occluder_w))
    final_y = max(0, min(final_y, mask.shape[0] - occluder_h))

    return int(final_x), int(final_y)

def compute_coverage_ratio(mask, occluder, x, y):
    """
    Compute the ratio of person pixels covered by the occluder.
    """
    person_pixels = (mask > 0).sum()
    if person_pixels == 0:
        return 0.0

    h, w = occluder.shape[:2]
    
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(mask.shape[1], x + w), min(mask.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return 0.0

    # Corresponding slice of the occluder
    ox0 = x0 - x
    oy0 = y0 - y
    ox1 = ox0 + (x1 - x0)
    oy1 = oy0 + (y1 - y0)

    mask_region = mask[y0:y1, x0:x1]
    occ_alpha = occluder[oy0:oy1, ox0:ox1, 3] > 0
    covered = ((mask_region > 0) & occ_alpha).sum()
    return covered / person_pixels

def select_occluder_for_video(occluder_imgs, occluder_types, frames, masks_path,
                              max_video_attempts=10):
    """
    Pick ONE occluder for the whole video. 
    For plates, also pick start/end y-positions defining a vertical trajectory that keeps coverage <= 0.6 on a few sampled frames. 
    For racks, pick a single fixed position.
    """
    if len(frames) == 0:
        return None

    sample_indices = sorted(set([0, len(frames) // 2, len(frames) - 1]))
    sampled = []
    for idx in sample_indices:
        frame_path = frames[idx]
        base_name = os.path.basename(frame_path)[:-4]
        mask = cv2.imread(os.path.join(masks_path, base_name + ".png"), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(frame_path)
        if mask is not None and img is not None:
            sampled.append((mask, img.shape[:2]))

    if not sampled:
        return None

    # Reference mask is used to size the occluder and place the anchor
    ref_mask, ref_shape = sampled[0]
    ys, xs = np.where(ref_mask > 0)
    if len(xs) == 0:
        # No person in the reference frame
        # Try to find another frame with a person as reference
        for m, s in sampled[1:]:
            ys, xs = np.where(m > 0)
            if len(xs) > 0:
                ref_mask, ref_shape = m, s
                break
        if len(xs) == 0:
            return None

    person_h = ys.max() - ys.min()
    img_h, img_w = ref_shape

    for _ in range(max_video_attempts):
        # Pick a random occluder
        idx = random.randrange(len(occluder_imgs))
        occluder_src = occluder_imgs[idx]
        occ_type = occluder_types[idx]

        # Scale the occluder relative to person height
        target_h = int(person_h * random.uniform(0.20, 0.40))
        if target_h <= 0:
            continue
        scale = target_h / occluder_src.shape[0]
        new_w = max(1, int(occluder_src.shape[1] * scale))
        new_h = max(1, int(occluder_src.shape[0] * scale))
        occluder = cv2.resize(occluder_src, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if occ_type == "rack":
            # Try a few anchors, keep the first that respects coverage
            placed = False
            for _ in range(5):
                x, y = get_anchor_position(ref_mask, new_h, new_w)
                # Check coverage on every sampled frame
                ok = all(compute_coverage_ratio(m, occluder, x, y) <= 0.6 for m, _ in sampled)
                if ok:
                    return {
                        "type": "rack",
                        "occluder": occluder,
                        "x": x,
                        "y": y,
                    }
            # Try another occluder
            continue

        else:
            # Anchor x stays the same, y goes from y_start to y_end across the video
            x_anchor, y_anchor = get_anchor_position(ref_mask, new_h, new_w)

            # Define a vertical travel range
            travel = int(person_h * random.uniform(0.6, 1.2))
            direction = random.choice([-1, 1])  # -1 = upward, 1 = downward

            y_start = y_anchor - (direction * travel // 2)
            y_end = y_anchor + (direction * travel // 2)

            # Clamp inside the image
            y_start = max(0, min(y_start, img_h - new_h))
            y_end = max(0, min(y_end, img_h - new_h))

            # Validate coverage on sampled frames
            n_sampled = len(sampled)
            ok = True
            for i, (m, _) in enumerate(sampled):
                t = i / max(1, n_sampled - 1)
                y_t = int(round(y_start + t * (y_end - y_start)))
                if compute_coverage_ratio(m, occluder, x_anchor, y_t) > 0.6:
                    ok = False
                    break
            if ok:
                return {
                    "type": "plate",
                    "occluder": occluder,
                    "x": x_anchor,
                    "y_start": y_start,
                    "y_end": y_end,
                }
            # otherwise try another occluder or configuration

    return None

def main(frames_path, masks_path, objects_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    frames = sorted(glob(os.path.join(frames_path, "*.png")))
    occluders = glob(os.path.join(objects_path, "*.png"))

    if not occluders:
        print("Error: No PNG files found in the objects directory.")
        return

    if not frames:
        print(f"Warning: No frames found in {frames_path}.")
        return

    # Load all occluders once
    loaded = []
    for p in occluders:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[2] != 4:
            continue
        loaded.append((img, get_occluder_type(p)))

    if not loaded:
        print("Error: No valid RGBA occluders.")
        return

    occluder_imgs = [o for o, _ in loaded]
    occluder_types = [t for _, t in loaded]

    # Pick ONE occluder and trajectory for the whole video
    config = select_occluder_for_video(occluder_imgs, occluder_types, frames, masks_path)

    if config is None:
        # No valid configuration found -> just copy frames as-is
        print(f"No valid occluder configuration found for {frames_path}, copying frames.")
        for frame_path in tqdm(frames):
            img = cv2.imread(frame_path)
            if img is not None:
                cv2.imwrite(os.path.join(output_path, os.path.basename(frame_path)), img)
        return

    n_frames = len(frames)

    for i, frame_path in enumerate(tqdm(frames)):
        img = cv2.imread(frame_path)
        if img is None:
            continue

        occluder = config["occluder"]

        if config["type"] == "rack":
            x, y = config["x"], config["y"]
        else:
            # Interpolate y between y_start and y_end over the video for a plate occluder
            t = i / max(1, n_frames - 1)
            x = config["x"]
            y = int(round(config["y_start"] + t * (config["y_end"] - config["y_start"])))

        result = overlay_transparent(img.copy(), occluder, x, y)
        cv2.imwrite(os.path.join(output_path, os.path.basename(frame_path)), result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='ArtificialOcclusion', description='Run artificial occlusion on a folder of frames')
    parser.add_argument('-video_filename', type=str, help='Name of the video file to run inference on')
    args = parser.parse_args()

    VIDEO_FILENAME = args.video_filename
    VIDEO_NAME = VIDEO_FILENAME[:-4]  # Remove file extension

    OBJECTS_DIR = os.path.join(ROOT_DIR, "data", "assets", "occluders")
    PARENT_FOLDER = os.path.join(ROOT_DIR, "data", "outputs", VIDEO_NAME)

    for folder in os.listdir(PARENT_FOLDER):
        FRAMES_DIR = os.path.join(PARENT_FOLDER, folder, "frames_without_background")
        MASKS_DIR = os.path.join(PARENT_FOLDER, folder, "sam3_filtered")
        OUT_DIR = os.path.join(PARENT_FOLDER, folder, "occluded_frames_without_background")

        main(
            frames_path=FRAMES_DIR,
            masks_path=MASKS_DIR,
            objects_path=OBJECTS_DIR,
            output_path=OUT_DIR
        )