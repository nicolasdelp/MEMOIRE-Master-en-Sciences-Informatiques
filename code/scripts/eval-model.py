import torch
import numpy as np
import lpips
import json
from pathlib import Path
from PIL import Image
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
from scipy import stats


DATASET_DIR = Path("/media/pc/hdd2/data-students/nicolasdelplanque/TrackMyLift-AI/data/outputs")

TEST_VIDEOS = {
    "squat": "IMG_0026",
    "bench": "IMG_0028",
    "deadlift": "IMG_0027",
}
AUGMENTATION = "original"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Init LPIPS

lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)

def load_image_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

def load_mask_tensor(path: Path) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    arr = np.array(mask, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0) > 0.5

def compute_metrics_clip(pred_dir: Path, gt_dir: Path, mask_dir: Path) -> dict:
    pred_files = sorted(pred_dir.glob("predicted_frame_*.png"))
    if not pred_files:
        raise FileNotFoundError(f"No frames found in {pred_dir}")

    ssim_scores, lpips_scores = [], []

    for pred_file in pred_files:
        idx_raw = pred_file.stem.split("_")[-1]
        idx5 = str(int(idx_raw)).zfill(5)

        gt_file = gt_dir / f"{idx5}.png"
        mask_file = mask_dir / f"{idx5}.png"

        if not gt_file.exists() or not mask_file.exists():
            print(f"Frame {idx5} missing (gt={gt_file.exists()}, mask={mask_file.exists()})")
            continue

        pred = load_image_tensor(pred_file).to(DEVICE)
        gt = load_image_tensor(gt_file).to(DEVICE)
        mask = load_mask_tensor(mask_file).to(DEVICE)

        occ_mask = ~mask # NOT operator (~)
        pred_masked = pred * occ_mask.float()
        gt_masked = gt * occ_mask.float()

        s = ssim_fn(pred_masked, gt_masked, data_range=1.0).item()
        ssim_scores.append(s)

        pred_lpips = pred_masked * 2 - 1
        gt_lpips = gt_masked * 2 - 1
        l = lpips_fn(pred_lpips, gt_lpips).item()
        lpips_scores.append(l)

    return {
        "ssim": float(np.mean(ssim_scores)),
        "lpips": float(np.mean(lpips_scores)),
        "ssim_per_frame": ssim_scores,
        "lpips_per_frame": lpips_scores,
        "n_frames": len(ssim_scores),
    }

results = {}

for movement, video in TEST_VIDEOS.items():
    print(f"\n=== {movement.upper()} ({video}) ===")
    results[movement] = {}

    gt_dir = DATASET_DIR / video / AUGMENTATION / "frames_without_background"
    mask_dir = DATASET_DIR / video / AUGMENTATION / "sam3_occluded"

    for model_name in ("baseline", "lora"):
        pred_dir = DATASET_DIR / video / AUGMENTATION / "taco" / model_name
        metrics = compute_metrics_clip(pred_dir, gt_dir, mask_dir)
        results[movement][model_name] = metrics
        print(f"  {model_name:8s} -> SSIM={metrics['ssim']:.4f}  "
              f"LPIPS={metrics['lpips']:.4f}  ({metrics['n_frames']} frames)")

# Wilcoxon test

print("\n=== WILCOXON TEST (all videos) ===")

all_ssim_base, all_ssim_lora = [], []
all_lpips_base, all_lpips_lora = [], []

for movement in results:
    all_ssim_base += results[movement]["baseline"]["ssim_per_frame"]
    all_ssim_lora += results[movement]["lora"]["ssim_per_frame"]
    all_lpips_base += results[movement]["baseline"]["lpips_per_frame"]
    all_lpips_lora += results[movement]["lora"]["lpips_per_frame"]

stat_ssim, p_ssim = stats.wilcoxon(all_ssim_lora,  all_ssim_base, alternative="greater")
stat_lpips, p_lpips = stats.wilcoxon(all_lpips_base, all_lpips_lora, alternative="greater")

print(f"  SSIM  : W={stat_ssim:.1f},  p={p_ssim:.4f}")
print(f"  LPIPS : W={stat_lpips:.1f}, p={p_lpips:.4f}")

# Export

with open("test_results.json", "w") as f:
    export = {
        mv: {
            model: {k: v for k, v in m.items() if not k.endswith("_per_frame")}
            for model, m in models.items()
        }
        for mv, models in results.items()
    }
    export["wilcoxon"] = {
        "ssim":  {"W": stat_ssim,  "p": p_ssim},
        "lpips": {"W": stat_lpips, "p": p_lpips},
    }
    json.dump(export, f, indent=2)
    print("\n[INFO] Résultats sauvegardés dans test_results.json")