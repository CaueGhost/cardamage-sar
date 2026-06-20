"""
maskrcnn_train.py — Treinamento e avaliação com Mask R-CNN
Treina o modelo e salva as predições de TODAS as imagens de validação.
Autor: Cauê Menezes — UNIFESP Laboratório 3
"""

import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

# ── Caminhos relativos ──────────────────────────────────────────────────────────
DIR  = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(DIR, "..", "COCO Car Damage (Projeto SAR)")
OUT  = os.path.join(DIR, "..", "resultados_maskrcnn")
os.makedirs(OUT, exist_ok=True)


class CarDamageDataset(Dataset):
    """Dataset de danos em carros no formato COCO."""

    def __init__(self, json_path, img_dir):
        with open(json_path) as f:
            data = json.load(f)
        self.img_dir = img_dir
        self.images  = data["images"]
        self.anns    = {}
        for ann in data["annotations"]:
            self.anns.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info   = self.images[idx]
        img        = Image.open(os.path.join(self.img_dir, img_info["file_name"])).convert("RGB")
        W, H       = img_info["width"], img_info["height"]
        img_tensor = TF.to_tensor(img)

        anns = self.anns.get(img_info["id"], [])
        boxes, labels, masks = [], [], []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(1)

            mask = np.zeros((H, W), dtype=np.uint8)
            seg  = ann.get("segmentation", [])
            if seg:
                import cv2
                pts = np.array(seg[0], dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], 1)
            else:
                mask[int(y):int(y + h), int(x):int(x + w)] = 1
            masks.append(torch.tensor(mask, dtype=torch.uint8))

        if boxes:
            target = {
                "boxes":  torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "masks":  torch.stack(masks),
            }
        else:
            target = {
                "boxes":  torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks":  torch.zeros((0, H, W), dtype=torch.uint8),
            }
        return img_tensor, target


# ── Datasets e DataLoaders ──────────────────────────────────────────────────────
train_ds = CarDamageDataset(
    os.path.join(BASE, "train", "COCO_train_annos.json"),
    os.path.join(BASE, "train"),
)
val_ds = CarDamageDataset(
    os.path.join(BASE, "val", "COCO_val_annos.json"),
    os.path.join(BASE, "val"),
)

collate = lambda x: tuple(zip(*x))
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,  collate_fn=collate)
val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, collate_fn=collate)

# ── Modelo ──────────────────────────────────────────────────────────────────────
model = maskrcnn_resnet50_fpn(weights="DEFAULT")
in_feat = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes=2)
in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, num_classes=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando: {device}")
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# ── Treino ──────────────────────────────────────────────────────────────────────
EPOCHS = 15
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"local {epoch+1}/{EPOCHS}", unit="batch")
    for imgs, targets in loop:
        imgs = [i.to(device) for i in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        loop.set_postfix(loss=f"{loss.item():.4f}")
    print(f"  → Loss médio: {total_loss / len(train_loader):.4f}")

# ── Salva o modelo ──────────────────────────────────────────────────────────────
save_path = os.path.join(DIR, "..", "maskrcnn_best.pt")
torch.save(model.state_dict(), save_path)
print(f"\nModelo salvo em: {save_path}")

# ── Avaliação em TODAS as imagens de validação ──────────────────────────────────
print("\nGerando predições para todas as imagens de validação...")

model.eval()
total_reais    = 0
total_detec    = 0
acertos        = 0

with open(os.path.join(BASE, "val", "COCO_val_annos.json")) as f:
    val_data = json.load(f)

anns_por_img = {}
for ann in val_data["annotations"]:
    anns_por_img.setdefault(ann["image_id"], []).append(ann)

    for img_info in tqdm(val_data["images"], desc="Avaliando", unit="img"):

        img_path = os.path.join(BASE, "img", img_info["file_name"])

        print("Arquivo:", img_info["file_name"])
        print("Caminho:", img_path)
        print("Existe?:", os.path.exists(img_path))

        if not os.path.exists(img_path):
            continue

    img_pil    = Image.open(img_path).convert("RGB")
    img_tensor = TF.to_tensor(img_pil).to(device)

    gt_boxes = [ann["bbox"] for ann in anns_por_img.get(img_info["id"], [])]
    n_reais  = len(gt_boxes)
    total_reais += n_reais

    with torch.no_grad():
        pred = model([img_tensor])[0]

    boxes_det  = [(b, s) for b, s in zip(pred["boxes"], pred["scores"]) if s > 0.5]
    masks_det  = [m for m, s in zip(pred["masks"], pred["scores"]) if s > 0.5]
    n_detec    = len(boxes_det)
    total_detec += n_detec
    acertos    += min(n_detec, n_reais)

    # ── Salva imagem com anotações ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(img_info["file_name"], fontsize=12, fontweight="bold")

    # Painel esquerdo — Ground truth
    axes[0].imshow(img_pil)
    axes[0].set_title(f"Ground Truth ({n_reais} danos)", color="green", fontweight="bold")
    axes[0].axis("off")
    for bbox in gt_boxes:
        x, y, w, h = bbox
        axes[0].add_patch(patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor="lime", facecolor="none"))

    # Painel direito — Predição Mask R-CNN
    axes[1].imshow(img_pil)
    axes[1].set_title(f"Mask R-CNN ({n_detec} detecções)", color="purple", fontweight="bold")
    axes[1].axis("off")
    for (box, score), mask_arr in zip(boxes_det, masks_det):
        x1, y1, x2, y2 = box.tolist()
        axes[1].add_patch(patches.Rectangle(
            (x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor="violet", facecolor="none"))
        axes[1].text(x1, y1 - 4, f"{score:.2f}", color="violet", fontsize=8, fontweight="bold")
        m = mask_arr[0].cpu().numpy()
        overlay = np.zeros((*m.shape, 4), dtype=float)
        overlay[m > 0.5] = [0.8, 0.1, 0.9, 0.35]
        axes[1].imshow(overlay)

    plt.tight_layout()
    nome_saida = os.path.join(OUT, f"pred_{img_info['file_name']}")
    plt.savefig(nome_saida, dpi=120, bbox_inches="tight")
    plt.close()

# ── Resumo final ────────────────────────────────────────────────────────────────
print("\n" + "=" * 45)
print("     RESULTADO FINAL — Mask R-CNN")
print("=" * 45)
print(f"  Imagens avaliadas    : {len(val_data['images'])}")
print(f"  Danos reais (total)  : {total_reais}")
print(f"  Detecções (conf>0.5) : {total_detec}")
print(f"  Acertos estimados    : {acertos}/{total_reais}")
print(f"  Taxa de acerto       : {acertos/total_reais*100:.1f}%")
print(f"  Imagens salvas em    : {OUT}")
print("=" * 45)