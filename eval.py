#!/usr/bin/env python3
# Tjrs gemini.
import os
import sys
import argparse
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Configuration Device
use_cuda = torch.cuda.is_available()
device = torch.device("mps" if False else ("cuda" if use_cuda else "cpu"))

# ==============================================================================
# 1. Fonctions d'Attaque (PGD Linf et PGD L2)
# ==============================================================================

def pgd_linf_attack(model, images, labels, epsilon=8/255, alpha=2/255, steps=20):
    """
    PGD Attack (L-infinity norm).
    Usual parameters for CIFAR-10: epsilon=8/255, alpha=2/255, steps=10 to 20.
    """
    images = images.to(device)
    labels = labels.to(device)
    
    # We start by adding a random noise
    delta = torch.zeros_like(images).uniform_(-epsilon, epsilon).to(device)
    delta = torch.clamp(images + delta, 0, 1) - images
    delta.requires_grad = True

    for _ in range(steps):
        output = model(images + delta)
        loss = F.cross_entropy(output, labels)

        # We check if the model is differentiable.
        try:
            loss.backward()
        except RuntimeError:
            pass
            
        if delta.grad is None:
            break

        grad = delta.grad.detach()
        
        # Gradient ascent step in the case of the L_inf norm.
        delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data = torch.clamp(images + delta, 0, 1) - images
        
        delta.grad.zero_()

    return (images + delta).detach()

def pgd_l2_attack(model, images, labels, epsilon=8/255, alpha=0.2, steps=20):
    """
    PGD Attack (L2 norm).
    Usual parameters for CIFAR-10 : epsilon=8/255, alpha=2/255, steps=10 to 20.
    """
    images = images.to(device)
    labels = labels.to(device)

    # We start with a small random noise.
    delta = torch.zeros_like(images).uniform_(-0.1, 0.1).to(device)
    delta.requires_grad = True

    for _ in range(steps):
        output = model(images + delta)
        loss = F.cross_entropy(output, labels)

        # We consider the cases with not differentiable model.
        try:
            loss.backward()
        except RuntimeError:
            pass

        if delta.grad is None:
            break

        grad = delta.grad.detach()
        
        # We normalize the gradient to compute the gradient ascent step with L2 norm.
        grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1) + 1e-10
        grad = grad / grad_norm.view(-1, 1, 1, 1)
        
        # Gradient ascent step in the case of the L2 norm.
        delta.data = delta + alpha * grad
        
        # Projection over the L2 ball.
        delta_norm = torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1)
        factor = torch.min(torch.ones_like(delta_norm), epsilon / (delta_norm + 1e-10))
        delta.data = delta * factor.view(-1, 1, 1, 1)
        
        # Clamp to have a valid image.
        delta.data = torch.clamp(images + delta, 0, 1) - images
        delta.grad.zero_()

    return (images + delta).detach()

# ==============================================================================
# 2. Utilitaires de Chargement (Copiés/Adaptés de test_project.py)
# ==============================================================================

def load_project(project_dir):
    module_filename = os.path.join(project_dir, 'model.py')
    if not os.path.exists(module_filename):
        raise FileNotFoundError(f"Could not find model.py in {project_dir}")

    sys.path = [project_dir] + sys.path
    spec = importlib.util.spec_from_file_location("model", module_filename)
    project_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(project_module)
    return project_module

def get_validation_loader(batch_size=64):
    # On prend les images brutes (ToTensor), le modèle gère le Canny en interne
    transform = transforms.ToTensor()
    cifar = torchvision.datasets.CIFAR10('./data/', train=False, download=True, transform=transform)
    # On prend tout le test set pour une vraie eval
    test_loader = torch.utils.data.DataLoader(cifar, batch_size=batch_size, shuffle=False, num_workers=2)
    return test_loader

# ==============================================================================
# 3. Fonction d'Evaluation Principale
# ==============================================================================

def evaluate(net, data_loader):
    net.eval() # Important : désactive Dropout/Batchnorm
    
    correct_nat = 0
    correct_linf = 0
    correct_l2 = 0
    total = 0
    
    print(f"Evaluation sur {device} en cours...")
    print(f"{'Batch':<10} | {'Nat Acc':<10} | {'PGD-Linf':<10} | {'PGD-L2':<10}")
    print("-" * 50)

    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        
        # 1. Accuracy Naturelle
        with torch.no_grad():
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            correct_nat += (predicted == labels).sum().item()

        # 2. Attaque PGD-Linf (Epsilon=8/255)
        # Note : On réactive les gradients pour l'attaque, même si net.eval() est actif
        with torch.enable_grad():
            adv_images_linf = pgd_linf_attack(net, images, labels, epsilon=8/255, alpha=2/255, steps=10)
        
        with torch.no_grad():
            outputs_linf = net(adv_images_linf)
            _, pred_linf = torch.max(outputs_linf.data, 1)
            correct_linf += (pred_linf == labels).sum().item()

        # 3. Attaque PGD-L2 (Epsilon=1.0)
        with torch.enable_grad():
            adv_images_l2 = pgd_l2_attack(net, images, labels, epsilon=8/255, alpha=0.2, steps=10)
            
        with torch.no_grad():
            outputs_l2 = net(adv_images_l2)
            _, pred_l2 = torch.max(outputs_l2.data, 1)
            correct_l2 += (pred_l2 == labels).sum().item()

        total += labels.size(0)
        
        if i % 10 == 0:
             print(f"{i:<10} | {100*correct_nat/total:.1f}%      | {100*correct_linf/total:.1f}%      | {100*correct_l2/total:.1f}%")

    print("-" * 50)
    print(f"Final Results (Total: {total} images):")
    print(f"Natural Accuracy : {100 * correct_nat / total:.2f}%")
    print(f"PGD-Linf Accuracy: {100 * correct_linf / total:.2f}% (eps=8/255)")
    print(f"PGD-L2 Accuracy  : {100 * correct_l2 / total:.2f}% (eps=1.0)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", metavar="project-dir", nargs="?", default=os.getcwd(),
                        help="Path to the project directory.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    
    args = parser.parse_args()
    
    try:
        project_module = load_project(args.project_dir)
        net = project_module.Net()
        net.to(device)
        
        # Chargement des poids
        # On utilise la méthode load_for_testing pour être sûr de charger comme le prof
        net.load_for_testing(project_dir=args.project_dir)
        print(f"Modèle chargé depuis {args.project_dir}")
        
    except Exception as e:
        print(f"Erreur chargement modèle: {e}")
        return

    test_loader = get_validation_loader(batch_size=args.batch_size)
    evaluate(net, test_loader)

if __name__ == "__main__":
    main()