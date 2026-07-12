# CIFAR-10 Image Classifier

A compact CIFAR-10 project built around a custom PyTorch CNN that reaches **91.01%** test accuracy. The repository is organized with a standard `src/` layout so the code is easier to browse and present on GitHub.

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **91.01%** |
| Dataset | CIFAR-10 (50k train / 10k test) |
| Training Epochs | 300 |
| Batch Size | 256 |

## Layout

```
├── src/
│   ├── CNN.py
│   ├── app.py
|   ├── main.ipynb
│   ├── main.py
|   ├── simplerunner.py
│   └── trained_net_91.01.pth
├── animal.jpg
├── uni.jpg
└── README.md
```

## What’s Included

- `src/CNN.py`: the configurable convolutional network
- `src/main.py`: training, evaluation, and sample inference
- `src/simplerunner.py`: a lightweight one-off inference script
- `src/app.py`: a Flask API for image classification
- `main.ipynb`: notebook version of the experiment
- `trained_net_91.01.pth`: the saved model weights

## Model Notes

The network uses repeated Conv2d → BatchNorm → ReLU → MaxPool blocks, followed by fully connected layers with dropout. The default training setup combines:

- Random crop, flip, rotation, affine translation, color jitter, and random erasing
- SGD with Nesterov momentum
- Weight decay and label smoothing
- Cosine annealing learning rate scheduling

## Getting Started

Install dependencies:

```bash
pip install torch torchvision flask requests pillow
```

If you want the DirectML path used during training, also install `torch-directml`.

Run the quick inference script:

```bash
python src/simplerunner.py
```

Run the Flask API:

```bash
python src/app.py
```

Retrain from scratch:

```bash
python src/main.py
```

The model expects 32×32 CIFAR-10 style inputs and predicts one of `plane`, `car`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, or `truck`.
