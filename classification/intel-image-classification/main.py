import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm, trange


class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            # input shape: (batch_size, 3, 224, 224) and
            # downsampled by a factor of 2^5 = 32 (5 times maxpooling)
            # So features' shape is (batch_size, 7, 7, 512)
            nn.Linear(in_features=7*7*512, out_features=4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # flatten
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_dataloaders(data_root, train_batch_size, test_batch_size):
    batch_size = {
        'train': train_batch_size,
        'test': test_batch_size,
    }
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # See: https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.ImageFolder
    intel_image_datasets = {
        phase: ImageFolder(root=os.path.join(data_root, f'seg_{phase}'), transform=data_transforms)
        for phase in ['train', 'test']
    }
    intel_image_dataloaders = {
        phase: DataLoader(intel_image_datasets[phase], batch_size=batch_size[phase], shuffle=(phase == 'train'))
        for phase in ['train', 'test']
    }
    return intel_image_dataloaders


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    training_loss = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), ascii=True):
        inputs, targets = inputs.to(device), targets.to(device)   # fetch data
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        training_loss += loss.item()

        # backward
        loss.backward()
        optimizer.step()

    training_loss /= len(train_loader.dataset)
    print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, training_loss))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, ascii=True):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test Accuracy: {:0.4f}\tLoss: {:.6f}'.format(float(correct) / len(test_loader.dataset), test_loss))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Intel-Image-Classification Example')
    parser.add_argument('--data-root', type=str, default='./intel-image-classification/',
                        help='path to dataset')
    parser.add_argument('--train-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='set GPU ID for CUDA training, -1 for CPU.')

    args = parser.parse_args()

    torch.initial_seed()
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() and args.gpu_id != -1 else 'cpu')

    dataloaders = get_dataloaders(args.data_root, args.train_batch_size, args.test_batch_size)
    model = VGG16(num_classes=6).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in trange(args.epochs):
        train(model, device, dataloaders['train'], optimizer, epoch)
        test(model, device, dataloaders['test'])

        torch.save(model.state_dict(), "model.pth")


if __name__ == '__main__':
    main()
