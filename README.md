# Detecção de Danos em Carros com Deep Learning
**Autor:** Cauê Menezes  
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

## Fundamentos teóricos

Para explicar **como** os modelos aprendem a detectar danos, vale destacar quatro conceitos centrais de deep learning presentes neste projeto:

### Redes neurais convolucionais (CNNs)

Os três modelos são redes neurais convolucionais. Cada camada extrai características visuais cada vez mais abstratas — bordas e texturas nas primeiras camadas, formas e padrões de dano nas camadas finais. O Faster R-CNN e o Mask R-CNN usam o backbone **ResNet50**; o YOLOv8 usa seu próprio backbone (CSPDarknet).

### Gradiente descendente estocástico (SGD)

O algoritmo de otimização usado para ajustar os pesos da rede durante o treino é o **SGD** (`torch.optim.SGD`). A cada batch de imagens, ele calcula o gradiente do erro (loss) em relação aos pesos e os atualiza na direção que reduz esse erro — daí o nome "descendente". É "estocástico" porque usa apenas um subconjunto (batch) dos dados a cada passo, não o dataset inteiro.

### Hiperparâmetros

São os valores definidos **antes** do treino, que controlam como a rede aprende (diferente dos pesos, que são aprendidos automaticamente):

| Hiperparâmetro | Valor usado | Função |
|---|---|---|
| `lr` (learning rate) | 0.005 | Tamanho do passo do gradiente a cada atualização |
| `momentum` | 0.9 | Acelera a convergência, suaviza oscilações |
| `weight_decay` | 0.0005 | Penaliza pesos muito grandes (evita overfitting) |
| `batch_size` | 2 a 4 | Quantidade de imagens processadas por vez |
| `epochs` | 15 (50 no YOLOv8) | Número de passagens completas pelo dataset |
| `conf` (threshold) | 0.5 | Confiança mínima para considerar uma detecção válida |

### Space warping (deformação do espaço de features)

Cada camada de uma rede neural aplica uma transformação linear seguida de uma não-linearidade:

```
h = ReLU(Wx) = max(0, Wx)
```

Geometricamente, isso "deforma" o espaço dos dados. Uma forma clássica de visualizar esse efeito é em 2D: imagine pontos de duas classes organizados de forma que nenhuma linha reta os separe (por exemplo, uma classe formando um círculo em torno da outra). Após aplicar `W` (rotação/escala) e `ReLU` (que zera valores negativos, "dobrando" o espaço sobre os eixos), os mesmos pontos se reorganizam em uma configuração onde uma única linha reta consegue separá-los.

É exatamente esse princípio, repetido em muitas camadas, que permite que o ResNet50 transforme pixels de uma imagem (espaço de entrada complexo) em representações onde "região com dano" e "região sem dano" se tornam linearmente separáveis para o classificador final.

---

## Resultados

| Modelo | Detecções corretas (val) | Taxa de acerto | Loss final | Épocas | Tempo de execução* |
|---|---|---|---|---|---|
| YOLOv8 | 0/24 | 0% | — | 13 (early stop) | 4m41s |
| Faster R-CNN | 15/24 | 62.5% | 0.1066 | 15 | 51m22s |
| **Mask R-CNN** | **20/24** | **83.3%** | **0.2492** | **15** | 49m11s |

\* Tempo medido em CPU (Intel Xeon E5-2680 v4), do início do treino até a avaliação completa em todas as imagens de validação.

### Conclusão

O **Mask R-CNN** obteve o melhor desempenho, detectando 20 dos 24 danos reais na validação (83.3%). Isso se deve ao fato de o modelo aprender também a segmentar o contorno exato dos danos, o que força a rede a compreender melhor a forma dos amassados.

O **Faster R-CNN** ficou em segundo lugar com 62.5% de acerto, enquanto o **YOLOv8** não generalizou para a validação, possivelmente por ter parado muito cedo (época 3) devido ao dataset reduzido.

Em compensação, o **YOLOv8** é disparadamente o mais rápido dos três (menos de 5 minutos contra quase 1 hora dos outros dois) — o que evidencia o clássico trade-off entre velocidade e precisão em deep learning. Faster R-CNN e Mask R-CNN levam tempos parecidos entre si, então o Mask R-CNN venceu sem custo extra de tempo significativo, tornando-o a melhor escolha geral neste cenário.

> Todos os modelos foram treinados com apenas 59 imagens de treino — um dataset pequeno para deep learning. O objetivo do trabalho é a **comparação entre métodos**, não atingir precisão máxima.

---

## Observações técnicas

- Todos os modelos foram treinados em **CPU** (Intel Xeon E5-2680 v4 @ 2.40GHz)
- Faster R-CNN e Mask R-CNN utilizaram backbone **ResNet50 pré-treinado no COCO**
- YOLOv8 utilizou o modelo **YOLOv8n** (nano) pré-treinado
- Tempo médio de treino por modelo: ~15 minutos na CPU
