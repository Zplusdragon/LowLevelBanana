from pathlib import Path
import sys

import cv2
import torch
import pyiqa
import numpy as np
from scipy.ndimage import uniform_filter


from basicsr.metrics.niqe import calculate_niqe
from basicsr.metrics.nr_iqa import calculate_nima

PRED_DIR = Path("") 
CROP_BORDER = 0                    
CONVERT_TO = "y"                    


def load_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def to_tensor_rgb(img_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def main():
    pred_dir = PRED_DIR
    files = sorted([p for p in pred_dir.iterdir() if p.is_file()])
    if not files:
        print("No images found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    brisque_metric = pyiqa.create_metric("brisque", device=device)

    niqe_scores = []
    brisque_scores = []
    nima_scores = []

    for img_path in files:
        img = load_image(img_path)
        img_tensor = to_tensor_rgb(img, device)
        niqe_score = calculate_niqe(
            img, crop_border=CROP_BORDER, input_order="HWC", convert_to=CONVERT_TO
        )
        niqe_score_val = float(np.asarray(niqe_score).squeeze())
        brisque_score = brisque_metric(img_tensor) * -1.0  
        brisque_score_val = float(brisque_score.detach().cpu().item())
        nima_score = calculate_nima(img_tensor)
        nima_score_val = float(nima_score)

        niqe_scores.append(niqe_score_val)
        brisque_scores.append(brisque_score_val)
        nima_scores.append(nima_score_val)

    print("-" * 50)
    print(f"Average NIQE: {sum(niqe_scores) / len(niqe_scores):.4f}")
    print(f"Average BRISQUE: {sum(brisque_scores) / len(brisque_scores):.4f}")
    print(f"Average NIMA: {sum(nima_scores) / len(nima_scores):.4f}")


if __name__ == "__main__":
    main()

