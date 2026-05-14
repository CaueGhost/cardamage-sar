# Detecção de Danos em Carros com Deep Learning
**Autor:** Cauê Eberspächer Menezes  
**Disciplina:** Visão Computacional — UNIFESP  
**Linguagem:** Python 3.14

---

## Objetivo

Comparar três métodos de deep learning para detectar danos (amassados) em carros, utilizando bases de imagens anotadas. Para cada método, são analisados quantas imagens foram corretamente detectadas e quantas foram erradas.

---

## Datasets utilizados

| Dataset | Imagens | Formato |
|---|---|---|
| COCO Car Damage | 59 treino / 11 validação | COCO JSON + JPG |
| CarDD | ~4.000 imagens | COCO JSON + JPG |

> ⚠️ Os datasets **não estão incluídos** neste repositório por questões de licença.  
> Baixe o COCO Car Damage em: https://github.com/datacluster-labs/car-damage-dataset  
> Baixe o CarDD em: https://github.com/CarDD-USTC/CarDD-USTC.github.io

---

## Modelos comparados

| # | Modelo | Biblioteca | Característica |
|---|---|---|---|
| 1 | **YOLOv8** | Ultralytics | Rápido, uma etapa, ideal para tempo real |
| 2 | **Faster R-CNN** | TorchVision | Duas etapas, alta precisão, clássico |
| 3 | **Mask R-CNN** | TorchVision | Detecção + segmentação de instâncias |

---

## Instalação

**Requisitos:** Python 3.10+ e pip instalados.

```bash
pip install -r requirements.txt
```

---

## Como usar

### 1. Exploração do dataset
```bash
python datareading.py
```

### 2. Treinar YOLOv8
```bash
python yolotraining.py
```

### 3. Treinar Faster R-CNN
```bash
python fasterrcnn_train.py
```

### 4. Treinar Mask R-CNN
```bash
python maskrcnn_train.py
```

---

## Estrutura do projeto

```
CODIGOSAR/
├── datareading.py          # Exploração e visualização do dataset
├── yolotraining.py         # Treinamento com YOLOv8
├── fasterrcnn_train.py     # Treinamento com Faster R-CNN
├── maskrcnn_train.py       # Treinamento com Mask R-CNN
├── requirements.txt        # Dependências do projeto
└── README.md               # Este arquivo
```

---

## Resultados

| Modelo | Detecções corretas (val) | Taxa de acerto | Loss final | Épocas |
|---|---|---|---|---|
| YOLOv8 | 0/24 | 0% | — | 13 (early stop) |
| Faster R-CNN | 15/24 | 62.5% | 0.1066 | 15 |
| **Mask R-CNN** | **20/24** | **83.3%** | **0.2492** | **15** |

### Conclusão

O **Mask R-CNN** obteve o melhor desempenho, detectando 20 dos 24 danos reais na validação (83.3%). Isso se deve ao fato de o modelo aprender também a segmentar o contorno exato dos danos, o que força a rede a compreender melhor a forma dos amassados.

O **Faster R-CNN** ficou em segundo lugar com 62.5% de acerto, enquanto o **YOLOv8** não generalizou para a validação, possivelmente por ter parado muito cedo (época 3) devido ao dataset reduzido.

> Todos os modelos foram treinados com apenas 59 imagens de treino — um dataset pequeno para deep learning. O objetivo do trabalho é a **comparação entre métodos**, não atingir precisão máxima.

---

## Observações técnicas

- Todos os modelos foram treinados em **CPU** (Intel Xeon E5-2680 v4 @ 2.40GHz)
- Faster R-CNN e Mask R-CNN utilizaram backbone **ResNet50 pré-treinado no COCO**
- YOLOv8 utilizou o modelo **YOLOv8n** (nano) pré-treinado
- Tempo médio de treino por modelo: ~15 minutos na CPU
