#!/usr/bin/env python3 
import os
import argparse
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import ssl

# Import transformations
from transformations.canye_edges import CannyEdge
from transformations.mean_shift import MeanShift
from transformations.thermometer import ThermometerEncoding

torch.manual_seed(42)
use_cuda = torch.cuda.is_available()
device = torch.device("mps" if False else ("cuda" if use_cuda else "cpu"))
ssl._create_default_https_context = ssl._create_unverified_context

valid_size = 1024 
batch_size = 64

class Net(nn.Module):
    model_file="models/default_model.pth"
    
    def __init__(self, simplification_layer=None):
        super().__init__()
        
        # Default to Canny if no layer provided (backward compatibility)
        if simplification_layer is None:
            self.simplification_layer = CannyEdge(low_threshold=10, high_threshold=75, blur=True)
        else:
            self.simplification_layer = simplification_layer
        
        if isinstance(self.simplification_layer, CannyEdge):
            in_channels = 1
        elif isinstance(self.simplification_layer, ThermometerEncoding):
            in_channels = 3 * self.simplification_layer.levels
        else:
            in_channels = 3
        
        self.backbone = torchvision.models.resnet18(weights=None)
        
        # Modify the first convolution to accept the correct number of channels
        self.backbone.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        
        # Output layer for CIFAR-10 (10 classes)
        # ResNet18 default fc is (512, 1000), we need (512, 10)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 10)

    def forward(self, x):
        x = self.simplification_layer(x)
        x = self.backbone(x)
        
        return F.log_softmax(x, dim=1)
    
    def save(self, model_file):
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        torch.save(self.state_dict(), model_file)

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file, map_location=device))

    def load_for_testing(self, project_dir='./'):
        self.load(os.path.join(project_dir, Net.model_file))
        self.eval()


def train_model(net, train_loader, pth_filename, num_epochs):
    print(f"Starting training on {device}")
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        epoch_loss = 0.0
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
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 200 == 199:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

        acc = 100 * correct / total
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        print(f"End of Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | Train Acc: {acc:.2f}%")

        if epoch % 10 == 0 or acc > best_acc:
            best_acc = acc
            net.save(pth_filename)
            print(f"-> Model saved (Acc: {acc:.2f}%)")

    net.save(pth_filename)
    print('Final model saved.')

def test_natural(net, test_loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i,data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def get_train_loader(dataset, valid_size=1024, batch_size=32):
    indices = list(range(len(dataset)))
    train_sampler = torch.utils.data.SubsetRandomSampler(indices[valid_size:])
    train = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)
    return train

def get_validation_loader(dataset, valid_size=1024, batch_size=32):
    indices = list(range(len(dataset)))
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[:valid_size])
    valid = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size)
    return valid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default=Net.model_file,
                        help="Name of the file used to load or to sore the model weights.")
    parser.add_argument('-f', '--force-train', action="store_true",
                        help="Force training even if model file already exists.")
    parser.add_argument('-e', '--num-epochs', type=int, default=20,
                        help="Set the number of epochs during training")
    args = parser.parse_args()

    # Default Net uses Canny
    net = Net()
    net.to(device)

    if not os.path.exists(args.model_file) or args.force_train:
        print(f"Training model to {args.model_file}")

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor() 
        ])
        
        cifar = torchvision.datasets.CIFAR10('./data/', train=True, download=True, transform=train_transform)
        train_loader = get_train_loader(cifar, valid_size, batch_size=batch_size)
        
        train_model(net, train_loader, args.model_file, args.num_epochs)
        print("Model save to '{}'.".format(args.model_file))

    print("Testing with model from '{}'. ".format(args.model_file))

    cifar_val = torchvision.datasets.CIFAR10('./data/', train=True, download=True, transform=transforms.ToTensor()) 
    valid_loader = get_validation_loader(cifar_val, valid_size)

    net.load(args.model_file)

    acc = test_natural(net, valid_loader)
    print("Model natural accuracy (valid): {:.2f}%".format(acc))

if __name__ == "__main__":
    main()