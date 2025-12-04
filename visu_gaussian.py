import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import kornia.filters as kfilters
import torchvision.transforms as transforms

# Configuration Device
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if False else "cpu"))

def tensor_to_numpy(tensor):
    img = tensor.cpu().detach().squeeze()
    if img.ndim == 3:
        img = img.permute(1, 2, 0)
    return img.numpy()

def visualize_gaussian_blur(
    target_class_name='airplane',
    sigmas=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    output_filename="visu_gaussian.png"
):
    print(f"Visualisation du flou Gaussien sur la classe '{target_class_name}'...")
    
    # 1. Charger une image de la classe cible
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    classes = dataset.classes
    target_idx = classes.index(target_class_name)
    
    img_tensor = None
    
    for img, label in dataset:
        if label == target_idx:
            img_tensor = img.unsqueeze(0).to(device)
            break
            
    if img_tensor is None:
        print(f"Classe {target_class_name} non trouvée.")
        return

    # 2. Boucle sur les sigmas
    cols = len(sigmas)
    
    fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3.5))
    plt.subplots_adjust(wspace=0.1)
    
    for i, sigma in enumerate(sigmas):
        print(f"  Applying Gaussian Blur with sigma = {sigma:.1f}...")
        
        if sigma == 0:
            blurred_img = img_tensor
        else:
            # Kernel size doit être impair. On prend une taille suffisante pour le sigma.
            # Règle empirique : kernel_size >= 6*sigma + 1
            k_size = int(2 * int(4.0 * sigma + 0.5) + 1)
            blurred_img = kfilters.gaussian_blur2d(img_tensor, (k_size, k_size), (sigma, sigma))
            
        # --- PLOTTING ---
        ax = axes[i]
        ax.imshow(tensor_to_numpy(blurred_img))
        
        if sigma == 0:
            ax.set_title("Original", fontweight='bold')
        else:
            ax.set_title(rf"$\sigma = {sigma:.1f}$", fontweight='bold')
        
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.axis('off')

    plt.suptitle(f"Gaussian kernel ({target_class_name})", fontsize=16, y=0.95)
    plt.savefig(output_filename, bbox_inches='tight', dpi=100)
    print(f"Sauvegardé sous : {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    visualize_gaussian_blur()
