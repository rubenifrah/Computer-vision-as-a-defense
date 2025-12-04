import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torchvision.transforms as transforms

# Importation des modules locaux
sys.path.append(os.getcwd())
from transformations.canye_edges import CannyEdge
from model import Net
from eval import pgd_linf_attack

# Configuration Device
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu"))

def tensor_to_numpy_img(tensor):
    """Convertit un tenseur en array numpy affichable."""
    img = tensor.cpu().detach().squeeze()
    if img.ndim == 2: return img.numpy()
    if img.ndim == 3: return img.permute(1, 2, 0).numpy()
    return img.numpy()

def get_one_sample_per_class():
    """Récupère exactement une image pour chaque classe (0-9)."""
    print("Chargement du dataset...")
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    classes = dataset.classes
    
    samples = {} 
    found_count = 0
    
    # On parcourt jusqu'à trouver les 10 classes
    for img, label in dataset:
        if label not in samples:
            samples[label] = img.unsqueeze(0).to(device)
            found_count += 1
        if found_count == 10:
            break
            
    # Retourne une liste triée [Image_Classe_0, Image_Classe_1, ...]
    return [samples[i] for i in range(10)], classes

def generate_summary_image(low, high, blur):
    # 1. Préparation
    output_dir = "visualisation_canny"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"summary_L{low}_H{high}_B{int(blur)}.png")
    
    print(f"Génération du résumé avec : Low={low}, High={high}, Blur={blur}")

    # 2. Chargement Modèle & Données
    # On utilise Net avec Canny
    canny_layer = CannyEdge(low_threshold=low, high_threshold=high, blur=blur)
    model = Net(simplification_layer=canny_layer).to(device)
    
    if os.path.exists(Net.model_file):
        model.load_for_testing()
    
    model.eval()
    
    samples, class_names = get_one_sample_per_class()
    
    # 3. Configuration du Plot (10 lignes, 4 colonnes)
    # Cols: [Orig RGB] [Orig Canny] [Attaque RGB] [Attaque Canny]
    rows = 10
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 20))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    # Paramètres de l'attaque pour la visualisation (Standard PGD Linf)
    linf_eps = 8/255
    linf_steps = 20

    print("Calcul des attaques et rendu graphique...")

    for i in range(rows):
        img_tensor = samples[i]
        label = torch.tensor([i]).to(device)
        class_name = class_names[i]
        
        # A. Génération de l'attaque (White Box sur le modèle actuel)
        with torch.enable_grad():
            adv_img = pgd_linf_attack(model, img_tensor, label, epsilon=linf_eps, steps=linf_steps)
            
        # B. Passage dans Canny (Vision du modèle)
        with torch.no_grad():
            edges_clean = model.simplification_layer(img_tensor)
            edges_adv = model.simplification_layer(adv_img)

        # C. Conversion pour affichage
        vis_clean_rgb = tensor_to_numpy_img(img_tensor)
        vis_clean_edge = tensor_to_numpy_img(edges_clean)
        vis_adv_rgb = tensor_to_numpy_img(adv_img)
        vis_adv_edge = tensor_to_numpy_img(edges_adv)

        # D. Remplissage du Plot
        # Col 1: Original RGB
        axes[i, 0].imshow(vis_clean_rgb)
        axes[i, 0].set_ylabel(class_name.upper(), fontweight='bold', fontsize=11, rotation=90)
        
        # Col 2: Original Canny
        axes[i, 1].imshow(vis_clean_edge, cmap='gray', vmin=0, vmax=1)
        
        # Col 3: Attacked RGB
        axes[i, 2].imshow(vis_adv_rgb)
        
        # Col 4: Attacked Canny
        axes[i, 3].imshow(vis_adv_edge, cmap='gray', vmin=0, vmax=1)

        # Suppression des axes (ticks) pour tout le monde
        for j in range(cols):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            # On retire le cadre noir pour faire plus propre
            for spine in axes[i, j].spines.values():
                spine.set_visible(False)

    # Titres des colonnes (Uniquement sur la première ligne)
    axes[0, 0].set_title("Original", fontweight='bold', fontsize=12)
    axes[0, 1].set_title(f"Canny Clean\n({low}-{high})", fontweight='bold', fontsize=12)
    axes[0, 2].set_title(f"Attaque PGD-Linf\n(eps={linf_eps:.3f})", fontweight='bold', fontsize=12, color='#D32F2F') # Rouge
    axes[0, 3].set_title("Canny Attaqué\n(Vision Modèle)", fontweight='bold', fontsize=12, color='#D32F2F')

    # Titre global
    plt.suptitle(f"Robustesse Visuelle Canny Edge (Paramètres: {low} / {high})", fontsize=16, y=0.90)
    
    # Sauvegarde
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"\n[Succès] Image de synthèse sauvegardée :")
    print(f"-> {os.path.abspath(filename)}")

if __name__ == "__main__":
    MY_CHOSEN_LOW = 10
    MY_CHOSEN_HIGH = 75
    MY_CHOSEN_BLUR = True
    
    try:
        generate_summary_image(MY_CHOSEN_LOW, MY_CHOSEN_HIGH, MY_CHOSEN_BLUR)
    except Exception as e:
        print(f"Erreur : {e}")
        import traceback
        traceback.print_exc()