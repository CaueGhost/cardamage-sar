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

Um dos principais avanços na Visão Computacional foi o desenvolvimento das Redes Neurais Convolucionais (CNNs), que superam as limitações dos classificadores lineares fully-connected. Como destacado por Johnson (2019, Lecture 7), ao contrário do procedimento de flatten (alongamento da imagem em um vetor unidimensional), que descarta a estrutura espacial dos dados, as CNNs operam diretamente sobre a disposição bidimensional dos pixels por meio de operações locais.
Os componentes essenciais incluem:

- Camadas de Convolução: Filtros (kernels) deslizam sobre a imagem, detectando padrões locais (bordas, texturas). Filtros sucessivos aumentam o receptive field.
- Funções de Ativação (ReLU): Introduzem não-linearidade e space warping.
- Pooling (Max Pooling): Reduz dimensionalidade e confere invariância a pequenas translações.
- Batch Normalization: Normaliza ativações por mini-batch, acelerando o treino e atuando como regularizador (Ioffe & Szegedy, 2015).

A arquitetura clássica segue [Conv + ReLU + Pool] × N seguido de camadas fully-connected. Exemplos como LeNet-5 (LeCun et al., 1998) e backbones modernos (ResNet50, CSPDarknet) utilizam esses princípios.
No presente projeto, os modelos YOLOv8, Faster R-CNN e Mask R-CNN aplicam esses conceitos para extrair características hierárquicas das imagens SAR, transformando pixels brutos em representações onde “amassado” e “sem dano” se tornam distinguíveis.

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

### Detecção (avaliação em todas as 24 anotações de dano da validação)

| Modelo | Detecções | Acertos | Precision | Recall (Taxa de acerto) | F1 | Épocas |
|---|---|---|---|---|---|---|
| YOLOv8 | 44 | 20/24 | 45.5% | **83.3%** | 58.8% | 22 (early stop) |
| Faster R-CNN | 13 | 13/24 | **100%** | 54.2% | **70.3%** | 15 |
| Mask R-CNN | 19 | 14/24 | 73.7% | 58.3% | 65.1% | 15 |

> **Precision** = Acertos / Detecções (quão confiável é cada alerta do modelo)
> **Recall** = Acertos / Danos reais (quantos danos o modelo realmente encontra)
> **F1** = média harmônica entre Precision e Recall (equilíbrio entre os dois)

### Conclusão

Não há um vencedor único — o melhor modelo depende da métrica priorizada:

- **Recall**: o **YOLOv8** vence (83.3%) — encontra a maior parte dos danos reais, ao custo de muitos falsos positivos (44 detecções para 24 danos reais).
- **Precision**: o **Faster R-CNN** vence (100%) — toda detecção que ele faz está correta, mas é conservador e deixa passar quase metade dos danos reais.
- **F1 (equilíbrio geral)**: o **Faster R-CNN** vence (70.3%) — o melhor compromisso entre não gerar alarmes falsos e ainda encontrar a maioria dos danos, seguido de perto pelo Mask R-CNN (65.1%).

Em um cenário prático, a escolha do modelo dependeria do objetivo do negócio: uma seguradora que não pode deixar nenhum dano passar sem inspeção priorizaria **Recall** (YOLOv8); uma oficina que quer evitar abrir ordens de serviço desnecessárias priorizaria **Precision** (Faster R-CNN), que neste caso também entrega o melhor equilíbrio geral (F1).

> Todos os modelos foram treinados com apenas 59 imagens de treino — um dataset pequeno para deep learning, o que explica a variabilidade dos resultados entre execuções (o treino de redes neurais tem componentes aleatórios, como a inicialização dos pesos). O objetivo do trabalho é a **comparação entre métodos**, não atingir precisão máxima.


---

### Referências
- Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. arXiv preprint arXiv:1502.03167.
- Johnson, J. (2019). CS231n: Convolutional Neural Networks for Visual Recognition [Lecture 7]. Stanford University / Michigan Online.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.

---

## Observações técnicas

- Os modelos foram treinados em **CPU**; o hardware específico pode variar entre execuções (na fase inicial, Intel Xeon E5-2680 v4, e para a fase final, Intel Core i7-13620H + NVIDIA Geforce RTX 4050) — ao comparar **tempo de treino** entre os três modelos, use medições feitas na mesma máquina para manter a comparação justa.
- Faster R-CNN e Mask R-CNN utilizaram backbone **ResNet50 pré-treinado no COCO**
- YOLOv8 utilizou o modelo **YOLOv8n** (nano) pré-treinado
- Tempo médio de treino por modelo: ~15 minutos na CPU
