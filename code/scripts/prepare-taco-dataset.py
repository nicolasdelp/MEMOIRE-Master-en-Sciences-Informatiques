import argparse
import json
import random
import re
import sys
from pathlib import Path
from collections import defaultdict


DIR_MAPPING = {
    "occluded_frames_without_background": ("occluded_frame", "occluded_frame"),
    "frames_without_background": ("amodal_object",  "amodal_object"),
    "sam3_occluded": ("visible_mask",   "visible_mask"),
}

AUGMENTATIONS = [
    "original", "blur", "brightness", "colorimetry",
    "contrast", "flipped", "flipped_colorimetry", "rotation",
]

# Video metadata
METADATA = {
    "IMG_0001": {"exercise": "squat",    "subject": "S01", "type": "clean"},
    "IMG_0002": {"exercise": "deadlift", "subject": "S01", "type": "clean"},
    "IMG_0003": {"exercise": "bench",    "subject": "S01", "type": "clean"},
    "IMG_0004": {"exercise": "squat",    "subject": "S02", "type": "clean"},
    "IMG_0005": {"exercise": "deadlift", "subject": "S02", "type": "clean"},
    "IMG_0006": {"exercise": "deadlift", "subject": "S02", "type": "clean"},
    "IMG_0007": {"exercise": "deadlift", "subject": "S03", "type": "clean"},
    "IMG_0008": {"exercise": "bench",    "subject": "S03", "type": "clean"},
    "IMG_0009": {"exercise": "bench",    "subject": "S03", "type": "real_occlusion"},
    "IMG_0010": {"exercise": "squat",    "subject": "S04", "type": "clean"},
    "IMG_0011": {"exercise": "squat",    "subject": "S03", "type": "real_occlusion"},
    "IMG_0012": {"exercise": "deadlift", "subject": "S04", "type": "clean"},
    "IMG_0013": {"exercise": "deadlift", "subject": "S04", "type": "clean"},
    "IMG_0014": {"exercise": "bench",    "subject": "S04", "type": "clean"},
    "IMG_0015": {"exercise": "squat",    "subject": "S05", "type": "clean"},
    "IMG_0016": {"exercise": "squat",    "subject": "S05", "type": "clean"},
    "IMG_0017": {"exercise": "deadlift", "subject": "S05", "type": "clean"},
    "IMG_0018": {"exercise": "deadlift", "subject": "S05", "type": "clean"},
    "IMG_0019": {"exercise": "deadlift", "subject": "S06", "type": "clean"},
    "IMG_0020": {"exercise": "squat",    "subject": "S07", "type": "clean"},
    "IMG_0021": {"exercise": "deadlift", "subject": "S07", "type": "clean"},
    "IMG_0022": {"exercise": "bench",    "subject": "S07", "type": "clean"},
    "IMG_0023": {"exercise": "deadlift", "subject": "S08", "type": "clean"},
    "IMG_0024": {"exercise": "squat",    "subject": "S08", "type": "clean"},
    "IMG_0025": {"exercise": "bench",    "subject": "S08", "type": "clean"},
    "IMG_0098": {"exercise": "squat",    "subject": "S03", "type": "real_occlusion"},
    "IMG_0099": {"exercise": "deadlift", "subject": "S03", "type": "real_occlusion"},
}

def extract_frame_number(filename: str) -> int:
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else -1

def reorganize_sequence(src_dir: Path, dst_dir: Path, dry_run: bool = False) -> dict:
    if not src_dir.exists():
        return {"ok": False, "msg": f"Dossier source inexistant : {src_dir}"}

    # Verify that the 3 source subfolders exist
    for src_name in DIR_MAPPING:
        if not (src_dir / src_name).exists():
            return {"ok": False, "msg": f"Missing sub-folder : {src_dir / src_name}"}

    ref_src = src_dir / "occluded_frames_without_background"
    ref_files = sorted(ref_src.glob("*.png"))
    if not ref_files:
        return {"ok": False, "msg": f"No PNG frames found in {ref_src}"}
    n_frames_ref = len(ref_files)

    # Create the destination folder
    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)

    # For each source subfolder, create the target subfolder and the links
    for src_subdir, (dst_subdir, file_prefix) in DIR_MAPPING.items():
        src_path = src_dir / src_subdir
        dst_path = dst_dir / dst_subdir

        if not dry_run:
            dst_path.mkdir(parents=True, exist_ok=True)

        src_files = sorted(src_path.glob("*.png"))
        if len(src_files) != n_frames_ref:
            return {
                "ok": False,
                "msg": f"Inconsistency in the number of frames : "
                       f"{src_path} has {len(src_files)} frames, "
                       f"expected {n_frames_ref}"
            }

        # Rename each frame to the expected format
        for i, src_file in enumerate(src_files):
            dst_file = dst_path / f"{file_prefix}_{i:03d}.png"
            if dry_run:
                continue
            
            if dst_file.exists() or dst_file.is_symlink():
                dst_file.unlink()
                
            dst_file.symlink_to(src_file.resolve())

    return {"ok": True, "n_frames": n_frames_ref, "msg": "OK"}


def split_videos(metadata: dict, train_frac: float, seed: int) -> tuple[list, list, list]:
    clean_videos = {v: m for v, m in metadata.items() if m.get("type", "clean") == "clean"}
    real_videos = sorted([v for v, m in metadata.items() if m.get("type") == "real_occlusion"])

    by_exercise = defaultdict(list)
    for video, meta in clean_videos.items():
        by_exercise[meta["exercise"]].append(video)

    rng = random.Random(seed)
    train_videos, val_videos = [], []

    for exercise, videos in by_exercise.items():
        videos = sorted(videos)
        rng.shuffle(videos)
        n = len(videos)
        n_train = max(1, round(n * train_frac))
        n_train = min(n_train, n - 1) if n > 1 else 1 # Ensure at least 1 valid value if possible
        train_videos.extend(videos[:n_train])
        val_videos.extend(videos[n_train:])
        print(f"  {exercise:10s}: train={n_train}, val={n - n_train} "
              f"(from {videos})")

    return sorted(train_videos), sorted(val_videos), real_videos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Root directory for raw data (IMG_XXXX/aug/...)")
    parser.add_argument("--output", required=True, type=Path, help="Root directory for output data (= dset_root for TACO)")
    parser.add_argument("--train-frac", type=float, default=0.70, help="Proportion of clean video frames")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Shows what would be done without creating anything")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"[ERROR] Missing input folder : {args.input}")

    print(f"[INFO] Input  : {args.input}")
    print(f"[INFO] Output : {args.output}")
    print(f"[INFO] {len(METADATA)} videos in METADATA "
          f"(with {sum(1 for m in METADATA.values() if m['type'] == 'clean')} clean)")
    print()

    # Split train/val
    print("[INFO] Split by source video (stratified by exercise) :")
    train_videos, val_videos, real_videos = split_videos(METADATA, args.train_frac, args.seed)
    print(f"\n  Train : {len(train_videos)} videos -> {train_videos}")
    print(f"  Val   : {len(val_videos)} videos -> {val_videos}")
    print(f"  Real  : {len(real_videos)} videos -> {real_videos} (non traitées, test qualitatif manuel)")

    train_subjects = set(METADATA[v]["subject"] for v in train_videos)
    val_subjects = set(METADATA[v]["subject"] for v in val_videos)
    overlap = train_subjects & val_subjects
    if overlap:
        print(f"\n[WARNING] Shared subjects: train/val : {sorted(overlap)}")

    train_paths, val_paths = [], []
    stats = {"ok": 0, "fail": 0}
    errors = []

    for video_list, json_list, label in [
        (train_videos, train_paths, "train"),
        (val_videos, val_paths,"val"),
    ]:
        for video in video_list:
            for aug in AUGMENTATIONS:
                src_dir = args.input / video / aug
                if not src_dir.exists():
                    continue
                dst_dir = args.output / video / aug
                result = reorganize_sequence(src_dir, dst_dir, dry_run=args.dry_run)
                if result["ok"]:
                    stats["ok"] += 1
                    json_list.append(f"{video}/{aug}")
                else:
                    stats["fail"] += 1
                    errors.append(f"[{label}] {video}/{aug} : {result['msg']}")

    print(f"\n[INFO] Reorganization : {stats['ok']} successes, {stats['fail']} failures")
    if errors:
        print(f"\n[INFO] First errors :")
        for err in errors[:10]:
            print(f"  - {err}")

    # JSON output
    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)
        train_json = args.output / "train.json"
        val_json = args.output / "val.json"

        with open(train_json, "w") as f:
            json.dump(train_paths, f, indent=2)
        with open(val_json, "w") as f:
            json.dump(val_paths, f, indent=2)

        print(f"\n[INFO] Written : {train_json} ({len(train_paths)} sequences)")
        print(f"[INFO] Written : {val_json}   ({len(val_paths)} sequences)")

        # Audit
        summary = {
            "seed": args.seed,
            "train_frac": args.train_frac,
            "train_videos": train_videos,
            "val_videos": val_videos,
            "real_videos_excluded": real_videos,
            "n_train_sequences": len(train_paths),
            "n_val_sequences": len(val_paths),
            "subject_overlap_train_val": sorted(overlap),
        }
        with open(args.output / "split_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[INFO] Audit : {args.output / 'split_summary.json'}")