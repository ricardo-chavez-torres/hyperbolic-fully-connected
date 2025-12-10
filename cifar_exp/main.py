from torchvision.datasets import CIFAR100
import torch
import torchvision
import random
import numpy as np
from pathlib import Path
import sys

from model import ourModel
parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir.parent))
from layers.lorentz import Lorentz


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(0)

dataset = CIFAR100(root='../data', train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                                      torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

manifold = Lorentz(k=0.1)

model = ourModel(manifold).double().cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

loss_fn = torch.nn.CrossEntropyLoss()
for epoch in range(200):
    running_loss = 0.0
    for idx, (images, labels) in enumerate(dataloader):
        images = images.cuda().double()
        labels = labels.cuda()
        optimizer.zero_grad()
        images = images.permute(0, 2, 3, 1)
        images = manifold.expmap0(images)
        images = images.permute(0, 3, 1, 2)


        output = model(images)
        loss = loss_fn(output, labels)
        
        loss.backward()
        optimizer.step()
        running_loss = running_loss * (idx / (idx + 1)) + loss.item() * (1 / (idx + 1))
        # if idx % 10 == 0 and idx > 0:
        #     print(f'Batch {idx}, Average Loss: {running_loss:.3f}, Accuracy: {(output.argmax(dim=1) == labels).float().mean().item() * 100:.2f}%')
    print(f'Epoch {epoch+1} completed. Average Loss: {running_loss:.3f}, Accuracy: {(output.argmax(dim=1) == labels).float().mean().item() * 100:.2f}%')