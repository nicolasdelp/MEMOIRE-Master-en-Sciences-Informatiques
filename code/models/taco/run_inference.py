import argparse
import os
import sys
import torch
import numpy as np

from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(FILE_DIR)))
TACO_DIR = os.path.join(ROOT_DIR, "code", "models", "taco")

sys.path.insert(0, TACO_DIR)
os.chdir(TACO_DIR)

from scripts.infer import VACDataset, collate_fn
from scripts import eval_utils
from sgm.util import instantiate_from_config
from omegaconf import OmegaConf


MODEL_PATHS = {
    "baseline": os.path.join(ROOT_DIR, "code", "checkpoints", "taco", "last.ckpt"),
    "lora":     os.path.join(ROOT_DIR, "code", "checkpoints", "taco", "lora.ckpt"),
}
CONFIG_PATH = os.path.join(TACO_DIR, "configs", "infer_vac.yaml")


def prepare_taco_input(video_name: str, dataset_dir: Path, taco_tmp_dir: Path) -> int:
    src_occluded = dataset_dir / video_name / "frames_without_background"
    src_mask = dataset_dir / video_name / "sam3_filtered"

    src_files = sorted(src_occluded.glob("*.png"))
    n_frames = len(src_files)
    assert n_frames > 0, f"Empty frame folder : {src_occluded}"

    dst_origin = taco_tmp_dir / video_name / "origin_frame"
    dst_mask = taco_tmp_dir / video_name / "visible_mask"
    dst_origin.mkdir(parents=True, exist_ok=True)
    dst_mask.mkdir(parents=True, exist_ok=True)

    for i, src_file in enumerate(src_files):
        dst = dst_origin / f"origin_frame_{i:03d}.png"
        if dst.exists() or dst.is_symlink():
            dst.unlink()  # Force a rebuild to avoid broken links
        dst.symlink_to(src_file.resolve())

    for i, src_file in enumerate(sorted(src_mask.glob("*.png"))):
        dst = dst_mask / f"mask_{i:03d}.png"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src_file.resolve())

    print(f"[INFO] Prepared TACO : {n_frames} frames in {taco_tmp_dir / video_name}")
    return n_frames

def build_windows(n_frames: int, window_size: int = 14, stride: int = 14) -> list[int]:
    windows = list(range(0, n_frames - window_size + 1, stride))

    # Add one final overlapping window if necessary
    last_start = n_frames - window_size
    if last_start not in windows and last_start >= 0:
        windows.append(last_start)

    return windows

FORCE_RESIZE = transforms.Compose([
    transforms.Resize((384, 384)),  # force exact, no aspect ratio preservation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def patched_getitem(self, idx, frame_start):
    video_path = os.path.join(self.dset_root, self.video_list[idx])
    clip_frames = np.arange(self.model_frames) + frame_start

    input_frames = []
    visible_masks = []
    for frame in clip_frames:
        # Use FORCE_RESIZE instead of self.image_transforms
        input_frames.append(FORCE_RESIZE(
            Image.open(os.path.join(
                video_path, 'origin_frame',
                f'origin_frame_{frame:03d}.png')).convert('RGB')))
        visible_masks.append(FORCE_RESIZE(
            Image.open(os.path.join(
                video_path, 'visible_mask',
                f'mask_{frame:03d}.png')).convert('RGB')))

    data_dict = self.construct_dict(input_frames, visible_masks, fps=6, motion_amount=127)
    data_dict['dset'] = torch.tensor([1])
    data_dict['idx'] = torch.tensor([idx])
    data_dict['frame_start'] = torch.tensor([frame_start])
    data_dict['clip_frames'] = torch.tensor(clip_frames)
    data_dict['video_id'] = self.video_list[idx]

    return data_dict

def run_taco_inference(video_name: str, model_key: str, taco_tmp_dir: Path, output_dir: Path, n_frames: int, num_samples: int = 10,  num_steps: int = 50):
    model_path = MODEL_PATHS[model_key]
    assert os.path.exists(model_path), f"Missing Checkpoint : {model_path}"

    # Charge config
    test_config = OmegaConf.load(CONFIG_PATH)
    test_config.model.params.conditioner_config.params.emb_models[0].params.open_clip_embedding_config.params.init_device = "cuda"
    test_config.model.params.ckpt_path = model_path
    test_config.model.params.use_ema = False
    test_config.model.params.ckpt_has_ema = False
    test_config.model.params.sampler_config.params.num_steps = num_steps
    test_config.model.params.sampler_config.params.guider_config.params.num_frames = 14
    test_config.model.params.sampler_config.params.guider_config.params.max_scale = 1.5
    test_config.model.params.sampler_config.params.guider_config.params.min_scale = 1.0
    test_config.model.params.sampler_config.params.device = "cuda"
    test_config.data.params.dset_root = str(taco_tmp_dir)

    print(f"\n[INFO] Loading model '{model_key}' from {model_path}...")
    with torch.device("cuda"):
        model = instantiate_from_config(test_config.model).to("cuda").eval()

    autocast_kwargs = eval_utils.prepare_model_inference_params(
        model, "cuda", num_steps, 14, 1.5, 1.0, autocast=1, decoding_t=14)

    # Load dataset
    data = VACDataset(
        dset_root=str(taco_tmp_dir),
        folder_name=video_name,
        train='val',
        frame_width=384, frame_height=384,
    )
    original_getitem = data.__class__.__getitem__

    # Output folder
    out_dir = output_dir / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_img = Image.open(taco_tmp_dir / video_name / "origin_frame" / "origin_frame_000.png")
    original_size = sample_img.size

    windows = build_windows(n_frames, window_size=14, stride=14)
    print(f"[INFO] {len(windows)} windows to process "
          f"(stride=14, last_start={windows[-1]}, n_frames={n_frames})")

    written_frames: set[int] = set()

    with torch.no_grad():
        with torch.autocast(**autocast_kwargs):

            for frame_start in tqdm(windows, desc=f"{model_key}/{video_name}"):

                def make_getitem(fs):
                    def _getitem(self, idx):
                        return patched_getitem(self, idx, fs)
                    return _getitem

                data.__class__.__getitem__ = make_getitem(frame_start)

                loader = DataLoader(
                    dataset=data, batch_size=1,
                    shuffle=False, collate_fn=collate_fn)

                for batch in loader:
                    batch = {k: v.to("cuda")
                             for k, v in batch.items()
                             if k not in ('video_id', 'num_video_frames')}
                    batch['num_video_frames'] = 14

                    # Average over num_samples to reduce noise
                    all_samples = []
                    for _ in range(num_samples):
                        video_dict = model.sample_video(
                            batch, enter_ema=False, limit_batch=False)
                        all_samples.append(
                            video_dict['sampled_video'].detach().cpu().numpy())

                    mean_frames = np.mean(all_samples, axis=0)  # (14, 3, H, W)

                    # Save with global index, without overwriting frames already written
                    for t in range(14):
                        global_idx = frame_start + t
                        if global_idx >= n_frames:
                            break
                        if global_idx in written_frames:
                            continue  # already written in a previous window

                        frame_hwc = np.transpose(mean_frames[t], (1, 2, 0))
                        img = Image.fromarray((frame_hwc * 255).astype(np.uint8))
                        img = img.resize(original_size, Image.LANCZOS)
                        img.save(out_dir / f"predicted_frame_{global_idx:03d}.png")
                        written_frames.add(global_idx)

    # Restore the original __getitem__
    data.__class__.__getitem__ = original_getitem

    n_written = len(written_frames)
    print(f"[INFO] {n_written}/{n_frames} frames saved in {out_dir}")
    assert n_written == n_frames, \
        f"[ERROR] {n_frames - n_written} frames missing !"

def main(video_name, output_path, model_key, num_samples, num_steps):
    dataset_dir = Path(os.path.join(ROOT_DIR, "data", "outputs"))
    taco_tmp_dir = Path(os.path.join(ROOT_DIR, "data", "taco_tmp"))

    n_frames = prepare_taco_input(video_name, dataset_dir, taco_tmp_dir)
    run_taco_inference(
        video_name=video_name,
        model_key=model_key,
        taco_tmp_dir=taco_tmp_dir,
        output_dir=output_path,
        n_frames=n_frames,
        num_samples=num_samples,
        num_steps=num_steps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run TACO inference')
    parser.add_argument('-video_filename', type=str, required=True)
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'lora'])
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--num_steps', type=int, default=50)
    args = parser.parse_args()

    VIDEO_NAME = args.video_filename[:-4]  # Remove file extension
    OUTPUT_PATH = Path(os.path.join(ROOT_DIR, "data", "outputs", VIDEO_NAME, "taco"))

    main(
        video_name=VIDEO_NAME,
        output_path=OUTPUT_PATH,
        model_key=args.model,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
    )