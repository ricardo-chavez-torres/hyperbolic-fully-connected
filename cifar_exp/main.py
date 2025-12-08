from torchvision.datasets import CIFAR100
import torch
import torchvision

from pathlib import Path
import sys

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from layers import Lorentz, Lorentz_Conv2d, Lorentz_fully_connected

# TODO: check effect of initialising bias to 0 vs. e.g. 0.01

dataset = CIFAR100(root='./data', train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                                      torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

manifold = Lorentz(k=1.0)
a = Lorentz_Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding="same", bias=True, manifold=manifold)

network = torch.nn.Sequential(
    Lorentz_Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding="same", bias=True, manifold=manifold),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    Lorentz_Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding="same", bias=True, manifold=manifold),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    Lorentz_Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding="same", bias=True, manifold=manifold),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    Lorentz_Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding="same", bias=True, manifold=manifold),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=4, stride=4),
    torch.nn.Flatten(),
    Lorentz_fully_connected(in_features=64, out_features=100, manifold=manifold, do_mlr=True),
).double().cuda()

optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
torch.manual_seed(0)
loss_fn = torch.nn.CrossEntropyLoss()
running_loss = 0.0
for epoch in range(50):
    for idx, (images, labels) in enumerate(dataloader):
        images = images.cuda().double()
        labels = labels.cuda()
        optimizer.zero_grad()
        images = images.permute(0, 2, 3, 1)
        images = manifold.expmap0(images)
        images = images.permute(0, 3, 1, 2)


        output = network(images)
        loss = loss_fn(output, labels)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # print(f'Loss: {loss.item()}')
        if idx % 10 == 0 and idx > 0:
            avg_loss = running_loss / 10
            print(f'Batch {idx}, Average Loss: {avg_loss:.3f}, Accuracy: {(output.argmax(dim=1) == labels).float().mean().item() * 100:.2f}%')
            running_loss = 0.0
    print(f'Epoch {epoch+1} completed.')