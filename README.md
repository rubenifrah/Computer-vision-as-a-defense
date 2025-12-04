# Computer Vision as a Defense: Robust CIFAR-10 Classifier with Canny Edge Detection

**Group "Jean Ponce"**: Gabriella FERNANDES MACEDO, Ruben IFRAH, Clément ROUVROY

## Project Overview

This project implements a robust classifier for the CIFAR-10 dataset, designed to defend against adversarial attacks (specifically PGD). The core idea is to use **Input Simplification** as a defense mechanism. By preprocessing images with a non-differentiable layer—specifically **Canny Edge Detection**—we remove high-frequency noise and textures that adversarial attacks typically exploit, forcing the model to rely on structural features.

Please read the rapport.pdf for more details.

### Key Features
- **Canny Edge Defense**: A custom PyTorch layer that applies Canny Edge detection to input images.
- **Gradient Obfuscation**: A "Straight-Through Estimator" trick is used to handle the non-differentiability of the Canny layer during backpropagation, effectively neutralizing gradient-based attacks in a white-box setting.
- **Alternative Defenses**: The codebase also includes implementations of other simplification layers like **Thermometer Encoding**, **Mean-Shift Filtering**, and **Quantization**.

## Methodology

### 1. Canny Edge Detection
Adversarial attacks often add imperceptible noise to textures. By converting images to binary edge maps, we destroy this fine-grained information. The model sees only the "skeleton" of the object, which is much harder to perturb without changing the semantic content of the image.

### 2. Handling Non-Differentiability
Since the Canny algorithm is non-differentiable, we cannot train the model end-to-end using standard backpropagation. We use a **Straight-Through Estimator (STE)**, which allows gradients to pass through the layer unchanged (or with a custom modification) during the backward pass.

> **Note**: While this technique provides high robustness against PGD attacks (Gradient Obfuscation), it may not be robust against adaptive attacks that approximate the gradients of the non-differentiable layer.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training
To train the robust model using Canny Edge detection:

```bash
./model_train.py --type canny --epochs 20
```

You can also explore other defenses:
```bash
./model_train.py --type meanshift
./model_train.py --type thermometer
```

### Evaluation
The training script automatically evaluates the model on Clean, PGD-Linf, and PGD-L2 accuracy at the end of each epoch.

To run a standalone test using the provided testing framework:
```bash
./test_project.py
```

### Visualization
To visualize the effect of the Canny Edge defense on adversarial examples:

```bash
python3 visu.py
```
This will generate a summary image in `visualisation_canny/` showing original images, their edge maps, and the corresponding attacked versions.

## Results

| Method | Clean Accuracy | PGD-Linf Accuracy |
|--------|----------------|-------------------|
| Standard ResNet | ~90% | ~0% |
| **Canny Edge Defense** | **~85%** | **~60%** |


## Project Structure

- `model.py`: Defines the `Net` class and the main model architecture.
- `model_train.py`: Main training script with support for different simplification layers.
- `transformations/`: Contains the implementation of defense layers:
    - `canye_edges.py`: Canny Edge detection layer.
    - `thermometer.py`: Thermometer encoding.
    - `mean_shift.py`: Mean-shift filtering.
    - `combo.py`: Combinations of filters.
- `rapport/`: Contains the LaTeX source of the project report.
