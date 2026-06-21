"""
datareading.py — Exploração e visualização do dataset COCO Car Damage
Exibe estatísticas do dataset e salva TODAS as imagens de validação
com os bounding boxes de dano marcados em vermelho.
Autor: Cauê Menezes — UNIFESP Laboratório 3
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm

# ── Caminhos relativos ──────────────────────────────────────────────────────────
DIR  = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(DIR, "..", "COCO Car Damage (Projeto SAR)")
OUT  = os.path.join(DIR, "..", "resultados_dataset")
os.makedirs(OUT, exist_ok=True)

# ── Carrega anotações de treino e validação ─────────────────────────────────────
with open(os.path.join(BASE, "train", "COCO_train_annos.json")) as f:
    train_data = json.load(f)

with open(os.path.join(BASE, "val", "COCO_val_annos.json")) as f:
    val_data = json.load(f)

# ── Estatísticas do dataset ─────────────────────────────────────────────────────
total_imgs = len(train_data["images"]) + len(val_data["images"])
total_anns = len(train_data["annotations"]) + len(val_data["annotations"])

print("=" * 45)
print("     ESTATÍSTICAS DO DATASET")
print("=" * 45)
print(f"  Imagens de treino    : {len(train_data['images'])}")
print(f"  Imagens de validação : {len(val_data['images'])}")
print(f"  Total de imagens     : {total_imgs}")
print(f"  Anotações de treino  : {len(train_data['annotations'])}")
print(f"  Anotações de val     : {len(val_data['annotations'])}")
print(f"  Total de anotações   : {total_anns}")
print(f"  Categorias           : {train_data['categories']}")
print(f"  Média de danos/img   : {total_anns / total_imgs:.1f}")
print("=" * 45)

# ── Mapa de anotações por imagem ────────────────────────────────────────────────
def mapear_anns(data):
    """Agrupa anotações pelo ID da imagem."""
    mapa = {}
    for ann in data["annotations"]:
        mapa.setdefault(ann["image_id"], []).append(ann)
    return mapa

train_anns = mapear_anns(train_data)
val_anns   = mapear_anns(val_data)

# ── Visualiza uma imagem de treino como exemplo ─────────────────────────────────
print("\nExibindo exemplo de imagem de treino...")

print("\nGerando visualizações de todas as imagens de treino...")

print("\nArquivos de validação:")
for img in val_data["images"]:
    print(img["file_name"])

print("\nArquivos de treino:")
for img in train_data["images"]:
    print(img["file_name"])

for img_info in tqdm(train_data["images"], desc="Treino", unit="img"):
    img_path = os.path.join(BASE, "train", img_info["file_name"])

    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path)
    anns = train_anns.get(img_info["id"], [])

    fig, ax = plt.subplots(1, figsize=(9, 9))
    ax.imshow(img)
    ax.axis("off")

    for ann in anns:
        x, y, w, h = ann["bbox"]
        ax.add_patch(
            patches.Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor="red",
                facecolor="none"
            )
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT, f"train_{img_info['file_name']}"),
        dpi=120,
        bbox_inches="tight"
    )
    plt.close()


