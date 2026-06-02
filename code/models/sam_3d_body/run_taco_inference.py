import argparse
import os
import json
import cv2
import numpy as np
import torch
from glob import glob
from typing import Any, Dict, List
from tqdm import tqdm

from tools.vis_utils import visualize_sample_together
from tools.build_fov_estimator import FOVEstimator
from sam_3d_body_estimator import SAM3DBodyEstimator
from build_models import load_sam_3d_body
from visualization.renderer import Renderer

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

def save_keypoints(outputs, save_dir, image_name):
    os.makedirs(save_dir, exist_ok=True)
    keypoints_data = []

    for pid, person_output in enumerate(outputs):
        person_kp = {
            "person_id": pid,
            "focal_length": float(person_output["focal_length"]),
            "bbox": person_output["bbox"].tolist(),
        }

        # 3D Keypoints
        if "pred_keypoints_3d" in person_output:
            person_kp["pred_keypoints_3d"] = person_output["pred_keypoints_3d"].tolist()

        # 2D Keypoints
        if "pred_keypoints_2d" in person_output:
            person_kp["pred_keypoints_2d"] = person_output["pred_keypoints_2d"].tolist()

        # Global joint coordinates
        if "pred_joint_coords" in person_output:
            person_kp["pred_joint_coords"] = person_output["pred_joint_coords"].tolist()

        # Total rotation per joint
        if "pred_global_rots" in person_output:
            person_kp["pred_global_rots"] = person_output["pred_global_rots"].tolist()

        # Camera movement
        if "pred_cam_t" in person_output:
            person_kp["pred_cam_t"] = person_output["pred_cam_t"].tolist()

        keypoints_data.append(person_kp)

    out_path = os.path.join(save_dir, f"{image_name}_keypoints.json")
    with open(out_path, "w") as f:
        json.dump(keypoints_data, f, indent=2)

def save_mesh_results(
    img_cv2: np.ndarray,
    outputs: List[Dict[str, Any]],
    faces: np.ndarray,
    save_dir: str,
    image_name: str,
) -> List[str]:
    """
    Save 3D mesh results to files and return PLY file paths
    """
    import json

    os.makedirs(save_dir, exist_ok=True)
    ply_files = []

    # Save focal length
    if outputs:
        focal_length_data = {"focal_length": float(outputs[0]["focal_length"])}
        focal_length_path = os.path.join(save_dir, f"{image_name}_focal_length.json")
        with open(focal_length_path, "w") as f:
            json.dump(focal_length_data, f, indent=2)

    for pid, person_output in enumerate(outputs):
        # Create renderer for this person
        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)

        # Store individual mesh
        tmesh = renderer.vertices_to_trimesh(
            person_output["pred_vertices"], person_output["pred_cam_t"], LIGHT_BLUE
        )
        mesh_filename = f"{image_name}_mesh_{pid:03d}.ply"
        mesh_path = os.path.join(save_dir, mesh_filename)
        tmesh.export(mesh_path)
        ply_files.append(mesh_path)

        # Save individual overlay image
        img_mesh_overlay = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                img_cv2.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)

        overlay_filename = f"{image_name}_overlay_{pid:03d}.png"
        cv2.imwrite(os.path.join(save_dir, overlay_filename), img_mesh_overlay)

        # Save bbox image
        img_bbox = img_cv2.copy()
        bbox = person_output["bbox"]
        img_bbox = cv2.rectangle(
            img_bbox,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),
            4,
        )
        bbox_filename = f"{image_name}_bbox_{pid:03d}.png"
        cv2.imwrite(os.path.join(save_dir, bbox_filename), img_bbox)

    return ply_files

def main(frame_folder_path, output_folder_path, checkpoint_path, fov_name, fov_path, mhr_path, bbox_thresh):
    # Initialize sam-3d-body model and other optional modules
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(checkpoint_path, device=device, mhr_path=mhr_path)
    
    fov_estimator = FOVEstimator(name=fov_name, device=device, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        fov_estimator=fov_estimator,
    )
    
    images_list = sorted(
        [
            image
            for image in glob(os.path.join(frame_folder_path, "*.png"))
        ]
    )

    for image_path in tqdm(images_list):
        base_name = os.path.basename(image_path)[:-4]
        os.makedirs(os.path.join(output_folder_path, base_name), exist_ok=True)
        outputs = estimator.process_one_image(
            image_path,
            bbox_thr=bbox_thresh,
            use_mask=False,
        )

        img = cv2.imread(image_path)
        rend_img = visualize_sample_together(img, outputs, estimator.faces)
        cv2.imwrite(
            os.path.join(output_folder_path, base_name, f"{base_name}.jpg"),
            rend_img.astype(np.uint8),
        )
        save_mesh_results(img, outputs, estimator.faces, os.path.join(output_folder_path, base_name), f"mesh_{base_name}")
        save_keypoints(outputs, os.path.join(output_folder_path, base_name), f"mesh_{base_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='SAM3DB', description='Run SAM-3D-Body inference on a folder of images')
    parser.add_argument('-video_filename', type=str, help='Name of the video file to run inference on')
    args = parser.parse_args()

    VIDEO_FILENAME = args.video_filename
    VIDEO_NAME = VIDEO_FILENAME[:-4] # Remove file extension

    CHECKPOINT_PATH = "/media/pc/hdd2/data-students/nicolasdelplanque/TrackMyLiftAI/models/sam-3d-body/checkpoints/v1.0.0/model.ckpt"
    MHR_PATH = "/media/pc/hdd2/data-students/nicolasdelplanque/TrackMyLiftAI/models/sam-3d-body/checkpoints/v1.0.0/mhr_model.pt"
    FOV_NAME = "moge2"
    FOV_PATH = "/media/pc/hdd2/data-students/nicolasdelplanque/TrackMyLiftAI/models/moge/checkpoints/v1.0.0/model.pt"
    BBOX_THRESH = 0.8
    
    FRAME_FOLDER_PATH = f"/media/pc/hdd2/data-students/nicolasdelplanque/TrackMyLift-AI/data/outputs/{VIDEO_NAME}/original/taco/lora/"
    OUTPUT_FOLDER_PATH = f"/media/pc/hdd2/data-students/nicolasdelplanque/TrackMyLift-AI/data/outputs/{VIDEO_NAME}/original/sam3dbody/lora/"

    main(
        frame_folder_path=FRAME_FOLDER_PATH,
        output_folder_path=OUTPUT_FOLDER_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        fov_name=FOV_NAME,
        fov_path=FOV_PATH,
        mhr_path=MHR_PATH,
        bbox_thresh=BBOX_THRESH,
    )
