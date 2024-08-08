import argparse
import torch
import numpy as np
import supervision as sv
from supervision.draw.color import ColorPalette
from supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import os

# 環境設定
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

class SAM2Predictor:
    def __init__(self, sam2_checkpoint, model_cfg, device):
        self.sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

    def set_image(self, image):
        self.sam2_predictor.set_image(np.array(image.convert("RGB")))

    def predict(self, input_boxes):
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False
        )

        if masks.ndim == 3:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)

        return masks, scores, logits

class DINODetector:
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

def process_keywords(keywords):
    return ' '.join([kw if kw.endswith('.') else kw + '.' for kw in keywords])

def get_image_files(images_dir):
    return [file for file in os.listdir(images_dir) 
            if file.lower().endswith((".jpg", ".jpeg"))]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", help="Directory path containing input images", required=True)
    parser.add_argument("--masks_dir", help="Directory path to save output masks", required=True)
    parser.add_argument("--keywords", nargs="+", help="Keywords for filtering images", required=True)
    args = parser.parse_args()

    images_dir = args.images_dir
    masks_dir = args.masks_dir
    keywords = process_keywords(args.keywords)

    sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    mask_predictor = SAM2Predictor(sam2_checkpoint, model_cfg, "cuda")

    model_id = "IDEA-Research/grounding-dino-tiny"
    dino_detector = DINODetector(model_id, "cuda")

    image_files = get_image_files(images_dir)

    for image_file in image_files:
        try:
            image_path = os.path.join(images_dir, image_file)
            image = Image.open(image_path)
            mask_predictor.set_image(image)

            input_boxes = dino_detector.detect_objects(image, keywords)
            masks, _, _ = mask_predictor.predict(input_boxes)

            masks = np.sum(masks, axis=0)
            masks = (masks > 0).astype(np.uint8)

            mask_path = os.path.join(masks_dir, f"{os.path.splitext(image_file)[0]}.png")
            Image.fromarray(masks * 255).save(mask_path)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

if __name__ == "__main__":
    main()