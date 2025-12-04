# GEMINI: permet de configurer les parametres pour Canny et Mean-Shift.
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
from transformations.mean_shift import MeanShift
from model import Net
from eval import pgd_linf_attack, pgd_l2_attack

# Configuration Device
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu"))

# --- CONFIG ATTAQUES ---
LINF_PARAMS = {'epsilon': 8/255, 'steps': 20}
L2_PARAMS   = {'epsilon': 1.0, 'steps': 20}

def tensor_to_numpy_img(tensor):
    img = tensor.cpu().detach().squeeze()
    if img.ndim == 2: return img.numpy()
    if img.ndim == 3: return img.permute(1, 2, 0).numpy()
    return img.numpy()

def get_samples_per_class():
    print("Extraction d'un exemple par classe...")
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    classes = dataset.classes
    samples = {} 
    found_count = 0
    for img, label in dataset:
        if label not in samples:
            samples[label] = img.unsqueeze(0).to(device)
            found_count += 1
        if found_count == 10: break
    return [samples[i] for i in range(10)], classes

def tune_canny(model, samples, class_names):
    current_low = 10
    current_high = 75
    use_blur = True
    
    print("\n" + "="*70)
    print("  MODE CALIBRAGE : CANNY EDGES")
    print("="*70)
    
    plt.ion()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    for i in range(10):
        img_tensor = samples[i]
        label = torch.tensor([i]).to(device)
        class_name = class_names[i]
        
        step_done = False
        while not step_done:
            print(f"\n--- CLASSE {i+1}/10 : {class_name.upper()} ---")
            print(f"Génération des attaques avec Canny({current_low}, {current_high})...")
            
            # Update Layer
            model.simplification_layer = CannyEdge(
                low_threshold=current_low, 
                high_threshold=current_high, 
                blur=use_blur
            ).to(device)

            # Attacks
            with torch.enable_grad():
                adv_linf = pgd_linf_attack(model, img_tensor, label, **LINF_PARAMS)
                adv_l2 = pgd_l2_attack(model, img_tensor, label, **L2_PARAMS)
            
            # Visualization
            with torch.no_grad():
                edges_clean = model.simplification_layer(img_tensor)
                edges_linf = model.simplification_layer(adv_linf)
                edges_l2 = model.simplification_layer(adv_l2)

            # Plotting
            axes[0, 0].imshow(tensor_to_numpy_img(img_tensor))
            axes[0, 0].set_title(f"Original ({class_name})")
            axes[0, 1].imshow(tensor_to_numpy_img(adv_linf))
            axes[0, 1].set_title("Attaque L-inf")
            axes[0, 2].imshow(tensor_to_numpy_img(adv_l2))
            axes[0, 2].set_title("Attaque L2")

            axes[1, 0].imshow(tensor_to_numpy_img(edges_clean), cmap='gray', vmin=0, vmax=1)
            axes[1, 0].set_title(f"Edges Clean\n(L:{current_low} H:{current_high})")
            axes[1, 1].imshow(tensor_to_numpy_img(edges_linf), cmap='gray', vmin=0, vmax=1)
            axes[1, 1].set_title("Vision sur L-inf")
            axes[1, 2].imshow(tensor_to_numpy_img(edges_l2), cmap='gray', vmin=0, vmax=1)
            axes[1, 2].set_title("Vision sur L2")

            for ax in axes.flatten(): ax.axis('off')
            plt.draw()
            plt.pause(0.1)

            # Interaction
            print("-" * 50)
            print("Analysez la ligne du BAS (Noir et Blanc).")
            print("\n[Q1] Bruit ? (y=Oui, monter Low / n=Non / l=Trop vide, baisser Low)")
            ans_noise = input("   Choix : ").lower()
            
            if ans_noise == 'y': current_low += 10
            elif ans_noise == 'l': current_low = max(5, current_low - 10)
            
            print("\n[Q2] Objet reconnaissable ? (y=Oui / n=Non, baisser High)")
            ans_struct = input("   Choix : ").lower()
            
            if ans_struct == 'n': current_high = max(current_low + 10, current_high - 15)
            
            if current_low >= current_high: current_high = current_low + 20

            print(f"\nNouveaux Params -> Low: {current_low} | High: {current_high}")
            res = input("Appuyez sur [Entrée] pour passer, 'r' pour re-tester : ").lower()
            if res != 'r': step_done = True

    plt.close()
    print(f"Configuration Canny Optimale : Low={current_low}, High={current_high}")

def tune_meanshift(model, samples, class_names):
    current_bw = 0.1
    current_iter = 5
    
    print("\n" + "="*70)
    print("  MODE CALIBRAGE : MEAN SHIFT")
    print("="*70)
    
    plt.ion()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    for i in range(10):
        img_tensor = samples[i]
        label = torch.tensor([i]).to(device)
        class_name = class_names[i]
        
        step_done = False
        while not step_done:
            print(f"\n--- CLASSE {i+1}/10 : {class_name.upper()} ---")
            print(f"Génération des attaques avec MeanShift(bw={current_bw:.2f}, iter={current_iter})...")
            
            # Update Layer
            model.simplification_layer = MeanShift(
                bandwidth=current_bw, 
                num_iterations=current_iter
            ).to(device)

            # Attacks
            with torch.enable_grad():
                adv_linf = pgd_linf_attack(model, img_tensor, label, **LINF_PARAMS)
                adv_l2 = pgd_l2_attack(model, img_tensor, label, **L2_PARAMS)
            
            # Visualization
            with torch.no_grad():
                out_clean = model.simplification_layer(img_tensor)
                out_linf = model.simplification_layer(adv_linf)
                out_l2 = model.simplification_layer(adv_l2)

            # Plotting
            axes[0, 0].imshow(tensor_to_numpy_img(img_tensor))
            axes[0, 0].set_title(f"Original ({class_name})")
            axes[0, 1].imshow(tensor_to_numpy_img(adv_linf))
            axes[0, 1].set_title("Attaque L-inf")
            axes[0, 2].imshow(tensor_to_numpy_img(adv_l2))
            axes[0, 2].set_title("Attaque L2")

            axes[1, 0].imshow(tensor_to_numpy_img(out_clean))
            axes[1, 0].set_title(f"MS Clean\n(bw:{current_bw:.2f} it:{current_iter})")
            axes[1, 1].imshow(tensor_to_numpy_img(out_linf))
            axes[1, 1].set_title("Vision sur L-inf")
            axes[1, 2].imshow(tensor_to_numpy_img(out_l2))
            axes[1, 2].set_title("Vision sur L2")

            for ax in axes.flatten(): ax.axis('off')
            plt.draw()
            plt.pause(0.1)

            # Interaction
            print("-" * 50)
            print("Analysez la ligne du BAS (Couleurs Aplaties).")
            print("\n[Q1] Bandwidth ? (+ = plus flou/abstrait, - = plus détaillé)")
            ans_bw = input("   (u = Up/Augmenter, d = Down/Diminuer, k = Keep) : ").lower()
            
            if ans_bw == 'u': current_bw += 0.05
            elif ans_bw == 'd': current_bw = max(0.01, current_bw - 0.05)
            
            print("\n[Q2] Iterations ? (Plus d'itérations = plus lisse)")
            ans_it = input("   (u = Up, d = Down, k = Keep) : ").lower()
            
            if ans_it == 'u': current_iter += 2
            elif ans_it == 'd': current_iter = max(1, current_iter - 2)

            print(f"\nNouveaux Params -> BW: {current_bw:.2f} | Iter: {current_iter}")
            res = input("Appuyez sur [Entrée] pour passer, 'r' pour re-tester : ").lower()
            if res != 'r': step_done = True

    plt.close()
    print(f"Configuration MeanShift Optimale : Bandwidth={current_bw:.2f}, Iterations={current_iter}")

def interactive_tuning():
    print("Chargement du modèle...")
    # On charge un modèle générique, peu importe les poids pour le calibrage de la couche d'entrée
    # car on regarde surtout la sortie de la couche de simplification.
    # MAIS pour les attaques, il faut un modèle qui a du sens, sinon les gradients sont nuls/aléatoires.
    # Idéalement il faudrait un modèle entraîné.
    
    model = Net().to(device)
    if os.path.exists(Net.model_file):
        model.load_for_testing()
        print("Poids chargés.")
    else:
        print("ATTENTION: Modèle non entraîné. Les attaques seront peut-être inefficaces.")

    model.eval()
    samples, class_names = get_samples_per_class()
    
    print("\nQuelle transformation voulez-vous calibrer ?")
    print("1. Canny Edge")
    print("2. Mean Shift")
    choice = input("Votre choix (1/2) : ")
    
    if choice == '1':
        tune_canny(model, samples, class_names)
    elif choice == '2':
        tune_meanshift(model, samples, class_names)
    else:
        print("Choix invalide.")

if __name__ == "__main__":
    interactive_tuning()