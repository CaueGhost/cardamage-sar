import json
import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF

BASE = r"C:\CODIGOSAR\COCO Car Damage (Projeto SAR)"

class CarDamageDataset(Dataset):
    def __init__(self, json_path, img_dir):
        with open(json_path) as f:
            data = json.load(f)
        self.img_dir = img_dir
        self.images = data['images']
        self.anns = {}
        for ann in data['annotations']:
            self.anns.setdefault(ann['image_id'], []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img = Image.open(os.path.join(self.img_dir, img_info['file_name'])).convert("RGB")
        img_tensor = TF.to_tensor(img)

        anns = self.anns.get(img_info['id'], [])
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
                labels.append(1)

        if boxes:
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64)
            }
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64)
            }
        return img_tensor, target

# Datasets
train_ds = CarDamageDataset(
    os.path.join(BASE, "train", "COCO_train_annos.json"),
    os.path.join(BASE, "train")
)
val_ds = CarDamageDataset(
    os.path.join(BASE, "val", "COCO_val_annos.json"),
    os.path.join(BASE, "val")
)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Modelo
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)  # fundo + damage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando: {device}")
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Treino
EPOCHS = 15
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, targets in train_loader:
        imgs = [i.to(device) for i in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Época {epoch+1}/{EPOCHS} — Loss: {total_loss/len(train_loader):.4f}")

# Salva o modelo
save_path = os.path.join(BASE, "..", "fasterrcnn_best.pt")
torch.save(model.state_dict(), save_path)
print(f"Modelo salvo em: {save_path}")

# Avaliação simples na validação
model.eval()
deteccoes, total = 0, 0
with torch.no_grad():
    for imgs, targets in val_loader:
        imgs = [i.to(device) for i in imgs]
        preds = model(imgs)
        for pred, tgt in zip(preds, targets):
            total += len(tgt['boxes'])
            deteccoes += len([s for s in pred['scores'] if s > 0.5])

print(f"\n=== Resultado Faster R-CNN ===")
print(f"Total de danos reais (val): {total}")
print(f"Total de detecções (conf > 0.5): {deteccoes}")