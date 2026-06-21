"""
comparacao.py — Comparação visual dos 3 modelos SAR
Exibe a mesma imagem de validação processada por YOLOv8, Faster R-CNN e Mask R-CNN lado a lado.
Autor: Cauê Menezes — UNIFESP Laboratório 3
"""

import json
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from ultralytics import YOLO

# ── Caminhos relativos (funciona em qualquer computador) ────────────────────────
DIR   = os.path.dirname(os.path.abspath(__file__))
BASE  = os.path.join(DIR, "..", "COCO Car Damage (Projeto SAR)")
MODS  = os.path.join(DIR, "..")


def carregar_faster_rcnn():
    """Carrega o Faster R-CNN treinado a partir do arquivo salvo."""
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes=2)
    model.load_state_dict(torch.load(os.path.join(MODS, "fasterrcnn_best.pt"), map_location="cpu"))
    model.eval()
    return model


def carregar_mask_rcnn():
    """Carrega o Mask R-CNN treinado a partir do arquivo salvo."""
    model = maskrcnn_resnet50_fpn(weights=None)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes=2)
    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, num_classes=2)
    model.load_state_dict(torch.load(os.path.join(MODS, "maskrcnn_best.pt"), map_location="cpu"))
    model.eval()
    return model


def carregar_yolo():
    """Localiza e carrega o melhor checkpoint YOLOv8 treinado."""
    runs_dir = os.path.join(DIR, "runs", "detect")
    runs = sorted(glob.glob(os.path.join(runs_dir, "*")))
    if not runs:
        raise FileNotFoundError("Nenhum treino YOLOv8 encontrado em runs/detect/")
    best_pt = os.path.join(runs[-1], "weights", "best.pt")
    return YOLO(best_pt)


# ── Carrega uma imagem de validação e o ground truth ────────────────────────────
val_json = os.path.join(BASE, "val", "COCO_val_annos.json")
with open(val_json) as f:
    val_data = json.load(f)

img_info = val_data["images"][0]
img_path = os.path.join(BASE, "val", img_info["file_name"])
img_pil  = Image.open(img_path).convert("RGB")
img_tensor = TF.to_tensor(img_pil)

gt_boxes = [ann["bbox"] for ann in val_data["annotations"]
            if ann["image_id"] == img_info["id"]]

# ── Roda os três modelos ────────────────────────────────────────────────────────
print("Carregando modelos...")
faster = carregar_faster_rcnn()
mask   = carregar_mask_rcnn()
yolo   = carregar_yolo()

print("Rodando predições...")
with torch.no_grad():
    faster_pred = faster([img_tensor])[0]
    mask_pred   = mask([img_tensor])[0]

yolo_pred = yolo.predict(img_path, conf=0.1, verbose=False)[0]

# ── Monta a figura comparativa ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
fig.suptitle(
    f"Comparação de modelos — {img_info['file_name']}",
    fontsize=14, fontweight="bold"
)

for ax in axes:
    ax.imshow(img_pil)
    ax.axis("off")

# Painel 1 — Ground truth
for bbox in gt_boxes:
    x, y, w, h = bbox
    axes[0].add_patch(patches.Rectangle(
        (x, y), w, h, linewidth=2, edgecolor="lime", facecolor="none"))
axes[0].set_title(f"Ground Truth ({len(gt_boxes)} danos)", fontweight="bold", color="green")

# Painel 2 — YOLOv8
count_y = sum(1 for b in yolo_pred.boxes if b.conf[0] > 0.1)
for b in yolo_pred.boxes:
    if b.conf[0] > 0.1:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        axes[1].add_patch(patches.Rectangle(
            (x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor="gold", facecolor="none"))
axes[1].set_title(f"YOLOv8 ({count_y} detecções)", fontweight="bold", color="goldenrod")

# Painel 3 — Faster R-CNN
count_f = 0
for box, score in zip(faster_pred["boxes"], faster_pred["scores"]):
    if score > 0.5:
        x1, y1, x2, y2 = box.tolist()
        axes[2].add_patch(patches.Rectangle(
            (x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor="deepskyblue", facecolor="none"))
        count_f += 1
axes[2].set_title(f"Faster R-CNN ({count_f} detecções)", fontweight="bold", color="steelblue")

# Painel 4 — Mask R-CNN com overlay de máscara
count_m = 0
for box, score, m_arr in zip(mask_pred["boxes"], mask_pred["scores"], mask_pred["masks"]):
    if score > 0.5:
        x1, y1, x2, y2 = box.tolist()
        axes[3].add_patch(patches.Rectangle(
            (x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor="violet", facecolor="none"))
        m = m_arr[0].numpy()
        overlay = np.zeros((*m.shape, 4), dtype=float)
        overlay[m > 0.5] = [0.8, 0.1, 0.9, 0.35]
        axes[3].imshow(overlay)
        count_m += 1
axes[3].set_title(f"Mask R-CNN ({count_m} detecções)", fontweight="bold", color="purple")

plt.tight_layout()

out = os.path.join(MODS, "comparacao_modelos.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nImagem salva em: {out}")
plt.show()
