import argparse
import cv2
import glob
import os
import torch
from tqdm import tqdm

from moge.utils.vis import colorize_depth
from moge.utils.io import write_depth
from moge.model.v2 import MoGeModel

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(FILE_DIR)))


def main(image_folder_path, output_path, visualisation_output_path=None, visualisation=False):
    os.makedirs(output_path, exist_ok=True) # Create output directory if it doesn't exist

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    model = MoGeModel.from_pretrained(pretrained_model_name_or_path=os.path.join(ROOT_DIR, "code", "checkpoints", "moge", "model.pt")).to(device)

    images_list = sorted(glob.glob(os.path.join(image_folder_path, "*.png")))

    os.makedirs(output_path, exist_ok=True)

    for image_path in tqdm(images_list, desc="Processing frames"):
        base_name = os.path.basename(image_path)[:-4] # Remove file extension
        raw_image = cv2.imread(image_path)
        
        # Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
        input_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)                       
        input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    

        # Infer 
        output = model.infer(input_image)
        depth_np = output["depth"].cpu().numpy()

        write_depth(os.path.join(output_path, f"{base_name}.png"), depth_np, max_range=1e5, compression_level=7)

        # Visualisation
        if visualisation and visualisation_output_path is not None:
            os.makedirs(visualisation_output_path, exist_ok=True)
            color_vis = colorize_depth(depth_np, cmap='magma')
            color_vis_bgr = cv2.cvtColor(color_vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(visualisation_output_path, f"{base_name}.png"), color_vis_bgr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='ExtractFrames', description='Extract all frames from a video into a folder')
    parser.add_argument('-video_filename', type=str, help='Name of the video file to extract frames from')
    parser.add_argument('-visualisation', type=bool, help='Whether to generate visualisation images', default=False)
    args = parser.parse_args()

    VIDEO_FILENAME = args.video_filename
    VISUALISATION = args.visualisation
    VIDEO_NAME = VIDEO_FILENAME[:-4] # Remove file extension

    PARENT_FOLDER = os.path.join(ROOT_DIR, "data", "outputs", VIDEO_NAME)
    
    for folder in os.listdir(PARENT_FOLDER):
        IMAGE_FOLDER_PATH = os.path.join(PARENT_FOLDER, folder, "frames")
        OUTPUT_PATH = os.path.join(PARENT_FOLDER, folder, "moge")
        VISUALISATION_OUTPUT_PATH = os.path.join(PARENT_FOLDER, folder, "moge_visualisation")

        main(
            image_folder_path=IMAGE_FOLDER_PATH, 
            output_path=OUTPUT_PATH,
            visualisation_output_path=VISUALISATION_OUTPUT_PATH,
            visualisation=VISUALISATION
        )

    
