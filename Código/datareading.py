import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

BASE = r"C:\CODIGOSAR\COCO Car Damage (Projeto SAR)"

# Carrega anotações
with open(os.path.join(BASE, "train", "COCO_train_annos.json")) as f:
    data = json.load(f)

print(f"Imagens: {len(data['images'])}")
print(f"Anotações: {len(data['annotations'])}")
print(f"Categorias: {data['categories']}")

# Visualiza uma imagem com bounding boxes
img_info = data['images'][0]
img = Image.open(os.path.join(BASE, "train", img_info['file_name']))

fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(img)

for ann in data['annotations']:
    if ann['image_id'] == img_info['id']:
        x, y, w, h = ann['bbox']
        rect = patches.Rectangle((x, y), w, h,
                                   linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

plt.title(f"Imagem: {img_info['file_name']} — danos marcados em vermelho")
plt.axis('off')
plt.show()