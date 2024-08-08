import argparse
import cv2
import torch
import numpy as np
import supervision as sv
from supervision.draw.color import ColorPalette
from supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import csv
import os

# environment settings
# use bfloat16
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# define SAM2ImagePredictor class
class SAM2Predictor():
    def __init__(self, sam2_checkpoint, model_cfg, device):
        self.sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

    
    def set_image(self, image):
        self.sam2_predictor.set_image(np.array(image.convert("RGB")))

    def predict(self, input_boxes):
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords = None,
            point_labels = None,
            box = input_boxes,
            multimask_output = False
        )

        # convert the shape to (n, H, W)
        if masks.ndim == 3:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)

        return masks, scores, logits

class DINODetector():
    def __init__(self, model_id, device):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

    def detect_objects(self, image, text):
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )

        input_boxes = results[0]["boxes"].cpu().numpy()

        return input_boxes
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", help="Directory path containing input images", required=True)
    parser.add_argument("--masks_dir", help="Directory path to save output masks", required=True)
    parser.add_argument("--keywords", nargs="+", help="Keywords for filtering images", required=True)
    args = parser.parse_args()

    # Access the values using args.images_dir and args.masks_dir
    images_dir = args.images_dir
    masks_dir = args.masks_dir
    keywords = args.keywords
    # Check if any keyword is missing a period at the end
    for i in range(len(keywords)):
        if not keywords[i].endswith('.'):
            keywords[i] += '.'

    # Join the keywords with spaces
    keywords = ' '.join(keywords)
    
    # build SAM2 image predictor
    sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    mask_predictor = SAM2Predictor(sam2_checkpoint, model_cfg, "cuda")

    # build grounding dino from huggingface
    model_id = "IDEA-Research/grounding-dino-tiny"

    dino_detector= DINODetector(model_id, "cuda")

    # Get the list of JPEG files in the images_dir
    image_files = [file for file in os.listdir(images_dir) 
                   if file.endswith(".jpg") or file.endswith(".jpeg") or
                   file.endswith(".JPG") or file.endswith(".JPEG")]

    # Loop through the image files
    for image_file in image_files:
        # Load the image
        image = Image.open(os.path.join(images_dir, image_file))

        mask_predictor.set_image(image)

        # get the box prompt for SAM 2
        input_boxes = dino_detector.detect_objects(image, keywords)

        masks, _, _ = mask_predictor.predict(input_boxes)

        masks = np.sum(masks, axis=0)
        masks = (masks > 0).astype(np.uint8)
        masks = np.repeat([masks * 255], 3, axis=0)
        masks = masks.transpose(1,2,0)

        cv2.imwrite(os.path.join(masks_dir, image_file), masks)    
    
if __name__ == "__main__":
    main()