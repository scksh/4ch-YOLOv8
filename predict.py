import argparse
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm


def load_image_pair(rgb_path, ir_path, img_size):
    rgb = cv2.imread(rgb_path)
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

    if rgb is None or ir is None:
        raise FileNotFoundError(f"Missing image: {rgb_path} or {ir_path}")

    rgb = cv2.resize(rgb, (img_size, img_size))
    ir = cv2.resize(ir, (img_size, img_size))

    ir = ir[..., np.newaxis]
    img_4ch = np.concatenate([rgb, ir], axis=-1)

    img_4ch = img_4ch.transpose((2, 0, 1))
    img_4ch = torch.from_numpy(img_4ch).float() / 255.0

    return img_4ch.unsqueeze(0)


def save_txt(result, save_path, img_shape):
    with open(save_path, 'w') as f:
        for box in result.boxes:
            cls = int(box.cls)
            conf = box.conf.item()
            xyxy = box.xyxy.cpu().numpy()[0]

            x1, y1, x2, y2 = xyxy
            x_c = (x1 + x2) / 2 / img_shape[1]
            y_c = (y1 + y2) / 2 / img_shape[0]
            w = (x2 - x1) / img_shape[1]
            h = (y2 - y1) / img_shape[0]

            f.write(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f} {conf:.4f}\n")


def run_inference(
    weights, source_rgb, source_ir, project, name,
    imgsz=640, conf=0.25, iou=0.45, max_det=100, device='cuda'
):
    save_dir = os.path.join(project, name)
    os.makedirs(save_dir, exist_ok=True)
    label_dir = os.path.join(save_dir, 'labels')
    os.makedirs(label_dir, exist_ok=True)

    model = YOLO(weights)
    model.fuse()
    model.to(device)

    rgb_paths = sorted(Path(source_rgb).glob("*"))
    ir_paths = sorted(Path(source_ir).glob("*"))

    assert len(rgb_paths) == len(ir_paths), "RGB and IR image counts must match"

    for rgb_path, ir_path in tqdm(zip(rgb_paths, ir_paths), total=len(rgb_paths)):
        rgb_path = str(rgb_path)
        ir_path = str(ir_path)

        img_tensor = load_image_pair(rgb_path, ir_path, imgsz).to(device)

        results = model(
            img_tensor,
            conf=conf,
            iou=iou,
            max_det=max_det,
            verbose=False
        )

        base_name = Path(rgb_path).stem
        txt_path = os.path.join(label_dir, f"{base_name}.txt")

        save_txt(results[0], txt_path, (imgsz, imgsz))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8-style 4-Channel Inference Script")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pt model weights")
    parser.add_argument("--source_rgb", type=str, required=True, help="Directory of RGB images")
    parser.add_argument("--source_ir", type=str, required=True, help="Directory of IR images")
    parser.add_argument("--project", type=str, default="runs/predict", help="Parent directory for results")
    parser.add_argument("--name", type=str, default="exp", help="Name of the current run")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (square)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--max_det", type=int, default=100, help="Max detections per image")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    args = parser.parse_args()

    run_inference(
        weights=args.weights,
        source_rgb=args.source_rgb,
        source_ir=args.source_ir,
        project=args.project,
        name=args.name,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device
    )
