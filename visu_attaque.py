# FICHIER GEMINI (flemme de faire l'affichage ; il le fait mieux que moi.)
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from PIL import Image
import torchvision.transforms as transforms

# Importation des modules locaux
sys.path.append(os.getcwd())

from transformations.canye_edges import CannyEdge
from transformations.mean_shift import MeanShift
from model import Net
from eval import pgd_linf_attack, pgd_l2_attack

# Configuration Device
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu"))

def tensor_to_pil(tensor):
    """Convertit un tenseur (C, H, W) en image PIL pour affichage/Canny."""
    tensor = tensor.cpu().detach().squeeze()
    if tensor.ndim == 3:
        tensor = tensor.permute(1, 2, 0) # CHW -> HWC
    np_img = (tensor.numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_img)

def tensor_to_numpy(tensor):
    img = tensor.cpu().detach().squeeze()
    if img.ndim == 3:
        img = img.permute(1, 2, 0)
    return img.numpy()

def visualize_attacks_and_layer(
    model, 
    layer_name,
    layer_params,
    linf_params, 
    l2_params, 
    output_filename="visu_attacks.png"
):
    """
    Génère une grille de comparaison : Original vs Attaques vs Vision du modèle.
    """
    
    # Setup Layer
    if layer_name == 'canny':
        low, high, use_blur = layer_params
        transform_layer = CannyEdge(low_threshold=low, high_threshold=high, blur=use_blur).to(device)
        cmap = 'gray'
    elif layer_name == 'meanshift':
        bw, it = layer_params
        transform_layer = MeanShift(bandwidth=bw, num_iterations=it).to(device)
        cmap = None # RGB
    
    # Inject into model for attacks
    model.simplification_layer = transform_layer
    
    print("Chargement des données...")
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    classes = dataset.classes
    
    # Sélectionner 5 images de classes différentes
    samples = []
    sample_labels = []
    found_classes = set()
    
    for img, label in dataset:
        if label not in found_classes:
            samples.append(img.unsqueeze(0)) 
            sample_labels.append(torch.tensor([label]))
            found_classes.add(label)
        if len(samples) >= 5: 
            break
            
    cols = 6 
    rows = len(samples)
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3 * rows))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    print("Génération des attaques et visualisation...")
    
    model.eval() 

    for idx, (img_tensor, label) in enumerate(zip(samples, sample_labels)):
        img_tensor = img_tensor.to(device)
        label = label.to(device)
        
        # 1. Attaques
        with torch.enable_grad():
            adv_linf = pgd_linf_attack(model, img_tensor, label, 
                                       epsilon=linf_params['epsilon'], 
                                       steps=linf_params['steps'])
            
            adv_l2 = pgd_l2_attack(model, img_tensor, label, 
                                   epsilon=l2_params['epsilon'], 
                                   steps=l2_params['steps'])

        # 2. Vision du modèle
        with torch.no_grad():
            view_orig = transform_layer(img_tensor)
            view_linf = transform_layer(adv_linf)
            view_l2 = transform_layer(adv_l2)
        
        # --- PLOTTING ---
        row_axes = axes[idx] if rows > 1 else axes
        class_name = classes[label.item()]
        
        # Original
        row_axes[0].imshow(tensor_to_numpy(img_tensor))
        row_axes[0].set_ylabel(class_name.upper(), fontweight='bold', fontsize=12)
        
        # Vision Original
        row_axes[1].imshow(tensor_to_numpy(view_orig), cmap=cmap)
        
        # Attaque Linf
        row_axes[2].imshow(tensor_to_numpy(adv_linf))
        
        # Vision Linf
        row_axes[3].imshow(tensor_to_numpy(view_linf), cmap=cmap)
        
        # Attaque L2
        row_axes[4].imshow(tensor_to_numpy(adv_l2))
        
        # Vision L2
        row_axes[5].imshow(tensor_to_numpy(view_l2), cmap=cmap)

        for ax in row_axes:
            ax.set_xticks([])
            ax.set_yticks([])

    # Titres
    axes[0, 0].set_title("Original", fontweight='bold')
    axes[0, 1].set_title(f"Vision ({layer_name})", fontweight='bold')
    axes[0, 2].set_title(f"PGD Linf\n(eps={linf_params['epsilon']:.3f})", fontweight='bold', color='red')
    axes[0, 3].set_title("Vision Modèle\n(post-Linf)", fontweight='bold')
    axes[0, 4].set_title(f"PGD L2\n(eps={l2_params['epsilon']:.1f})", fontweight='bold', color='red')
    axes[0, 5].set_title("Vision Modèle\n(post-L2)", fontweight='bold')

    plt.suptitle(f"Visualisation: Impact des Attaques sur {layer_name.upper()}", fontsize=16, y=0.95)
    plt.savefig(output_filename, bbox_inches='tight', dpi=100)
    print(f"Sauvegardé sous : {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    # ==========================================
    # CONFIGURATION
    # ==========================================
    
    LINF_CONFIG = {'epsilon': 8/255, 'steps': 20}
    L2_CONFIG = {'epsilon': 1.0, 'steps': 20}
    
    # Choix du layer
    print("Quel layer visualiser ?")
    print("1. Canny")
    print("2. Mean Shift")
    choice = input("Choix (1/2) : ")
    
    try:
        net = Net()
        net.to(device)
        if os.path.exists(Net.model_file):
            net.load_for_testing()
            print("Poids chargés.")
        
        if choice == '1':
            # Canny Params: (Low, High, Blur)
            PARAMS = (10, 75, True)
            visualize_attacks_and_layer(net, 'canny', PARAMS, LINF_CONFIG, L2_CONFIG, "visu_canny.png")
        elif choice == '2':
            # MeanShift Params: (Bandwidth, Iterations)
            PARAMS = (0.1, 5)
            visualize_attacks_and_layer(net, 'meanshift', PARAMS, LINF_CONFIG, L2_CONFIG, "visu_meanshift.png")
        else:
            print("Choix invalide.")
            
    except Exception as e:
        print(f"Erreur : {e}")
        import traceback
        traceback.print_exc()