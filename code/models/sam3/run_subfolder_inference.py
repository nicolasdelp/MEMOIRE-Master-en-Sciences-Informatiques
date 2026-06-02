import argparse
import os
import torch
import glob

from PIL import Image
from tqdm import tqdm

from model_builder import build_sam3_image_model
from model.sam3_image_processor import Sam3Processor
from visualization_utils import save_binary_mask


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(FILE_DIR)))

bpe_path = os.path.join(ROOT_DIR, "code", "checkpoints", "sam3", "bpe_simple_vocab_16e6.txt.gz")
checkpoint_path = os.path.join(ROOT_DIR, "code", "checkpoints", "sam3", "sam3.pt")

def main(image_folder_path, output_path):
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")

    model = build_sam3_image_model(
            bpe_path=bpe_path,
            device="cuda",
            eval_mode=True,
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
        )

    output_folder = os.path.join(output_path)

    os.makedirs(output_folder, exist_ok=True) # Create output directory if it doesn't exist

    images_list = sorted(glob.glob(os.path.join(image_folder_path, "*.png")))

    processor = Sam3Processor(model, confidence_threshold=0.5)

    for img_path in tqdm(images_list, desc="Processing frames"):
        image = Image.open(img_path)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = processor.set_image(image)
            prompt_text_str = "person"
            inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt_text_str)

        if inference_state["masks"] is not None:
            save_binary_mask(mask_tensor=inference_state["masks"], output_path=output_folder, image_name=img_path)
            
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='SAM3', description='Run SAM-3 inference on a folder of images')
    parser.add_argument('-video_filename', type=str, help='Name of the video file to run inference on')
    parser.add_argument('-occluded', type=bool, default=False, help='Whether to run inference on occluded frames or not')
    args = parser.parse_args()

    VIDEO_FILENAME = args.video_filename
    VIDEO_NAME = VIDEO_FILENAME[:-4] # Remove file extension
    OCCLUDED = args.occluded
    PARENT_FOLDER = os.path.join(ROOT_DIR, "data", "outputs", VIDEO_NAME)
    
    for folder in os.listdir(PARENT_FOLDER):
        if OCCLUDED:
            IMAGE_FOLDER_PATH = os.path.join(PARENT_FOLDER, folder, "occluded_frames_without_background")
            OUTPUT_PATH = os.path.join(PARENT_FOLDER, folder, "sam3_occluded")
        else:
            IMAGE_FOLDER_PATH = os.path.join(PARENT_FOLDER, folder, "frames")
            OUTPUT_PATH = os.path.join(PARENT_FOLDER, folder, "sam3")

        main(
            image_folder_path=IMAGE_FOLDER_PATH, 
            output_path=OUTPUT_PATH,
        )
