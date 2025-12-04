#!/usr/bin/env python3

import os, os.path, sys
import argparse
import importlib 
import importlib.abc
import torch, torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import ssl

torch.manual_seed(42)
use_cuda = torch.cuda.is_available()
device = torch.device("mps" if torch.mps.is_available() else ("cuda" if use_cuda else "cpu"))
ssl._create_default_https_context = ssl._create_unverified_context

def load_project(project_dir):
    module_filename = os.path.join(project_dir, 'model.py')
    if os.path.exists(project_dir) and os.path.isdir(project_dir) and os.path.isfile(module_filename):
        print(f"Loaded '{project_dir}' on {device}.")
    else:
        print(f"'{project_dir}' not known.")
        raise FileNotFoundError 

    sys.path = [project_dir] + sys.path
    spec = importlib.util.spec_from_file_location("model", module_filename)
    project_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(project_module)

    return project_module

def train(net, train_loader, criterion, optimizer, epoch):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}] Loss: {round(avg_loss,3)} | Acc: {round(acc,2)}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", metavar="project-dir", nargs="?", default=os.getcwd(),
                        help="Chemin vers le dossier du projet.")
    parser.add_argument("-b", "--batch-size", type=int, default=128,
                        help="Taille du batch.")
    parser.add_argument("-e", "--epochs", type=int, default=500,
                        help="Nombre d'epochs.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001).")

    args = parser.parse_args()

    project_module = load_project(args.project_dir)
    net = project_module.Net()
    net.to(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale() # vous pouvez changer ici la transfo, je vais tenter canye edge soon tm.
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    print(f"Running for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train(net, train_loader, criterion, optimizer, epoch)
    print("Traiing finished.")
    save_path = os.path.join(args.project_dir, 'final_weights.pth')
    torch.save(net.state_dict(), save_path)
    print(f"Save on {save_path}")

if __name__ == "__main__":
    main()