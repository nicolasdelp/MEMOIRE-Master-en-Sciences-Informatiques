import json
import numpy as np
from pathlib import Path


ROOT_DIR = Path("/media/pc/hdd2/data-students/nicolasdelplanque/TrackMyLift-AI")

TEST_VIDEOS = {
    "squat": "IMG_0026",
    "bench": "IMG_0027",
    "deadlift": "IMG_0028",
}

CONDITIONS = ["occulted", "baseline", "lora"]


def find_keypoints_file(frame_dir: Path) -> Path | None:
    candidates = list(frame_dir.glob("*_keypoints.json"))
    return candidates[0] if candidates else None

def get_frame_index(folder_name: str) -> int | None:
    digits = ''.join(filter(str.isdigit, folder_name))
    return int(digits) if digits else None

def load_condition_keypoints(condition_dir: Path) -> dict[int, np.ndarray]:
    kp_by_frame = {}
    for frame_dir in sorted(condition_dir.iterdir()):
        if not frame_dir.is_dir():
            continue
        idx = get_frame_index(frame_dir.name)
        if idx is None:
            continue
        kp_file = find_keypoints_file(frame_dir)
        if kp_file is None:
            continue
        with open(kp_file) as f:
            data = json.load(f)
        if not data:
            continue
        kp = data[0].get("pred_keypoints_3d")
        if kp is not None:
            kp_by_frame[idx] = np.array(kp)
    return kp_by_frame

def align_root(kp: np.ndarray, root_idx: int = 0) -> np.ndarray:
    return kp - kp[root_idx]

def mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred - gt, axis=-1)))

def compute_mpjpe_video(video_name: str, condition: str) -> dict:
    base_dir = ROOT_DIR / "data" / "outputs" / video_name / "original" / "sam3dbody"
    gt_dir = base_dir / "ground_truth"
    pred_dir = base_dir / condition

    if not gt_dir.exists():
        print(f"  [WARN] GT not found : {gt_dir}")
        return {}
    if not pred_dir.exists():
        print(f"  [WARN] Condition not found : {pred_dir}")
        return {}

    gt_kps = load_condition_keypoints(gt_dir)
    pred_kps = load_condition_keypoints(pred_dir)

    # Frames common to both conditions
    common_frames = sorted(set(gt_kps.keys()) & set(pred_kps.keys()))
    if not common_frames:
        print(f"  [WARN] No common frames found between GT and {condition}")
        return {}

    errors = []
    for idx in common_frames:
        gt_kp = align_root(gt_kps[idx])
        pred_kp = align_root(pred_kps[idx])
        if gt_kp.shape != pred_kp.shape:
            continue
        errors.append(mpjpe(pred_kp, gt_kp))

    if not errors:
        return {}

    return {
        "mean_mpjpe_mm": float(np.mean(errors) * 1000),
        "std_mpjpe_mm": float(np.std(errors)  * 1000),
        "n_frames": len(errors),
        "per_frame": errors,
    }


# Evaluation

results = {}

for movement, video in TEST_VIDEOS.items():
    print(f"\n=== {movement.upper()} ({video}) ===")
    results[movement] = {}

    for condition in CONDITIONS:
        r = compute_mpjpe_video(video, condition)
        if r:
            results[movement][condition] = r
            print(f"  {condition:10s} -> MPJPE = {r['mean_mpjpe_mm']:.1f} ± {r['std_mpjpe_mm']:.1f} mm  ({r['n_frames']} frames)")
        else:
            print(f"  {condition:10s} -> no data")


# Export

out_path = ROOT_DIR / "data" / "outputs" / "mpjpe_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[INFO] Results saved to {out_path}")

# Boxplot

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

movements_labels = {
    "squat": "Squat",
    "bench": "Développé couché",
    "deadlift": "Soulevé de terre",
}
conditions_labels = {
    "occulted": "Occlus",
    "baseline": "TACO-baseline",
    "lora": "TACO-LoRA",
}
colors = {
    "occulted": "#5B9BD5",
    "baseline": "#ED7D31",
    "lora": "#70AD47",
}

fig, axes = plt.subplots(1, 3, figsize=(14, 6))

for ax, (mov_key, mov_label) in zip(axes, movements_labels.items()):
    plot_data  = []
    box_colors = []
    xtick_labels = []

    for cond_key, cond_label in conditions_labels.items():
        if mov_key in results and cond_key in results[mov_key]:
            # meter to mm
            values = [v * 1000 for v in results[mov_key][cond_key]["per_frame"]]
            plot_data.append(values)
            box_colors.append(colors[cond_key])
            xtick_labels.append(cond_label)

    bp = ax.boxplot(
        plot_data,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker="o", markersize=2, alpha=0.3),
        widths=0.5,
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_title(mov_label, fontsize=12, fontweight="bold")
    ax.set_ylabel("MPJPE (mm)" if ax == axes[0] else "")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(xtick_labels, fontsize=9, rotation=15, ha="right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

patches = [mpatches.Patch(color=colors[k], label=conditions_labels[k], alpha=0.8)
           for k in conditions_labels]

plt.tight_layout()

boxplot_path = ROOT_DIR / "data" / "outputs" / "mpjpe_boxplot.png"
plt.savefig(boxplot_path, bbox_inches="tight", dpi=150)
print(f"[INFO] Boxplot saved to {boxplot_path}")