"""
experimento_externo.py — Avaliação de Generalização em Imagens Externas
Roda os 3 modelos já treinados (YOLOv8, Faster R-CNN, Mask R-CNN) sobre
imagens de qualquer fonte externa, sem necessidade de anotações.
Autor: Cauê Menezes — UNIFESP Laboratório 3
"""

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

# ── Configuração ────────────────────────────────────────────────────────────────
DIR  = os.path.dirname(os.path.abspath(__file__))
MODS = os.path.join(DIR, "..")
OUT  = os.path.join(DIR, "..", "resultados_externo")
os.makedirs(OUT, exist_ok=True)

# ── Pasta com imagens externas ──────────────────────────────────────────────────
# Coloque aqui o caminho da pasta com suas imagens externas.
# Aceita JPG, JPEG e PNG. Sem necessidade de anotações.
PASTA_IMAGENS = os.path.join(DIR, "..", "imagens_externas")

# ── Carregamento dos modelos ────────────────────────────────────────────────────
def carregar_faster_rcnn():
    """Carrega Faster R-CNN a partir do checkpoint salvo."""
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes=2)
    path = os.path.join(MODS, "fasterrcnn_best.pt")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def carregar_mask_rcnn():
    """Carrega Mask R-CNN a partir do checkpoint salvo."""
    model = maskrcnn_resnet50_fpn(weights=None)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes=2)
    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, num_classes=2)
    path = os.path.join(MODS, "maskrcnn_best.pt")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def carregar_yolo():
    """Localiza o melhor checkpoint YOLOv8 treinado."""
    runs_dir = os.path.join(DIR, "runs", "detect")
    runs = sorted(glob.glob(os.path.join(runs_dir, "*")))
    if not runs:
        raise FileNotFoundError("Nenhum treino YOLOv8 encontrado em runs/detect/")
    best_pt = os.path.join(runs[-1], "weights", "best.pt")
    return YOLO(best_pt)

# ── Coleta de imagens externas ──────────────────────────────────────────────────
extensoes = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
imagens = []
for ext in extensoes:
    imagens += glob.glob(os.path.join(PASTA_IMAGENS, ext))
imagens = sorted(imagens)

if not imagens:
    print(f"\n⚠️  Nenhuma imagem encontrada em: {PASTA_IMAGENS}")
    print("Crie a pasta e coloque suas imagens externas lá (JPG ou PNG).")
    print("Exemplo: C:\\CODIGOSAR\\imagens_externas\\foto1.jpg")
    exit(1)

print(f"\n{'='*50}")
print(f"  Experimento Externo — Avaliação de Generalização")
print(f"{'='*50}")
print(f"  Imagens encontradas: {len(imagens)}")
print(f"  Carregando modelos...")

faster = carregar_faster_rcnn()
mask   = carregar_mask_rcnn()
yolo   = carregar_yolo()
print("  Modelos carregados. Iniciando inferência...\n")

# ── Inferência e geração de resultados ─────────────────────────────────────────
resumo = []

for i, img_path in enumerate(imagens):
    nome = os.path.basename(img_path)
    print(f"  [{i+1}/{len(imagens)}] {nome}")

    img_pil    = Image.open(img_path).convert("RGB")
    img_tensor = TF.to_tensor(img_pil)

    # YOLOv8
    yolo_pred  = yolo.predict(img_path, conf=0.1, verbose=False)[0]
    yolo_boxes = [b for b in yolo_pred.boxes if b.conf[0] > 0.1]
    n_yolo     = len(yolo_boxes)

    # Faster R-CNN
    with torch.no_grad():
        faster_pred = faster([img_tensor])[0]
    faster_boxes = [(b, s) for b, s in zip(faster_pred["boxes"], faster_pred["scores"]) if s > 0.5]
    n_faster     = len(faster_boxes)

    # Mask R-CNN
    with torch.no_grad():
        mask_pred = mask([img_tensor])[0]
    mask_boxes = [(b, s, m) for b, s, m in zip(mask_pred["boxes"], mask_pred["scores"], mask_pred["masks"]) if s > 0.5]
    n_mask     = len(mask_boxes)

    resumo.append({ "nome": nome, "yolo": n_yolo, "faster": n_faster, "mask": n_mask })

    # ── Figura comparativa (4 painéis) ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(f"Imagem externa: {nome}", fontsize=12, fontweight="bold")

    for ax in axes:
        ax.imshow(img_pil)
        ax.axis("off")

    axes[0].set_title("Imagem original", fontweight="bold")

    # YOLOv8
    axes[1].set_title(f"YOLOv8  ({n_yolo} detecções)", color="goldenrod", fontweight="bold")
    for b in yolo_boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        conf = b.conf[0].item()
        axes[1].add_patch(patches.Rectangle(
            (x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor="gold", facecolor="none"))
        axes[1].text(x1, max(y1-4, 0), f"{conf:.2f}",
                     color="gold", fontsize=8, fontweight="bold")

    # Faster R-CNN
    axes[2].set_title(f"Faster R-CNN  ({n_faster} detecções)", color="steelblue", fontweight="bold")
    for box, score in faster_boxes:
        x1, y1, x2, y2 = box.tolist()
        axes[2].add_patch(patches.Rectangle(
            (x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor="deepskyblue", facecolor="none"))
        axes[2].text(x1, max(y1-4, 0), f"{score:.2f}",
                     color="deepskyblue", fontsize=8, fontweight="bold")

    # Mask R-CNN
    axes[3].set_title(f"Mask R-CNN  ({n_mask} detecções)", color="purple", fontweight="bold")
    for box, score, m_arr in mask_boxes:
        x1, y1, x2, y2 = box.tolist()
        axes[3].add_patch(patches.Rectangle(
            (x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor="violet", facecolor="none"))
        axes[3].text(x1, max(y1-4, 0), f"{score:.2f}",
                     color="violet", fontsize=8, fontweight="bold")
        m = m_arr[0].numpy()
        overlay = np.zeros((*m.shape, 4), dtype=float)
        overlay[m > 0.5] = [0.8, 0.1, 0.9, 0.35]
        axes[3].imshow(overlay)

    plt.tight_layout()
    saida = os.path.join(OUT, f"externo_{os.path.splitext(nome)[0]}.png")
    plt.savefig(saida, dpi=130, bbox_inches="tight")
    plt.close()

# ── Resumo final ────────────────────────────────────────────────────────────────
total_yolo   = sum(r["yolo"]   for r in resumo)
total_faster = sum(r["faster"] for r in resumo)
total_mask   = sum(r["mask"]   for r in resumo)

print(f"\n{'='*50}")
print(f"     RESULTADO FINAL — Experimento Externo")
print(f"{'='*50}")
print(f"  Imagens avaliadas        : {len(resumo)}")
print(f"  Detecções YOLOv8         : {total_yolo}")
print(f"  Detecções Faster R-CNN   : {total_faster}")
print(f"  Detecções Mask R-CNN     : {total_mask}")
print(f"  Resultados salvos em     : {OUT}")
print(f"{'='*50}")

# ── Salva resumo em txt ─────────────────────────────────────────────────────────
resumo_path = os.path.join(OUT, "resumo_externo.txt")
with open(resumo_path, "w", encoding="utf-8") as f:
    f.write("RESULTADO FINAL — Experimento Externo\n")
    f.write("=" * 50 + "\n")
    f.write(f"Imagens avaliadas        : {len(resumo)}\n")
    f.write(f"Detecções YOLOv8         : {total_yolo}\n")
    f.write(f"Detecções Faster R-CNN   : {total_faster}\n")
    f.write(f"Detecções Mask R-CNN     : {total_mask}\n\n")
    f.write("Detalhes por imagem:\n")
    f.write(f"{'Imagem':<30} {'YOLOv8':>8} {'Faster':>8} {'Mask':>8}\n")
    f.write("-" * 58 + "\n")
    for r in resumo:
        f.write(f"{r['nome']:<30} {r['yolo']:>8} {r['faster']:>8} {r['mask']:>8}\n")

print(f"\n  Resumo também salvo em: {resumo_path}")
