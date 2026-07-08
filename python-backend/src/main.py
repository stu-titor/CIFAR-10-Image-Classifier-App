from PIL import Image

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchvision
import torchvision.transforms as transforms
from CNN import ImageNeuralNetwork

# %%
import torch_directml
device = torch_directml.device()
print(f'Using device: {device}')

# %%
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),  
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
    transforms.ColorJitter(brightness=0.2, contrast=0.2), 
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# %%
train_data = torchvision.datasets.CIFAR10(root = './data', train = True, transform = train_transform, download = True)
test_data = torchvision.datasets.CIFAR10(root = './data', train = False, transform = test_transform, download = True)

train_loader = torch.utils.data.DataLoader(train_data, batch_size = 256, shuffle = True, num_workers = 4)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 256, shuffle = False, num_workers = 4)

# %%
image, label = train_data[0]
image.size()

# %%
class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# %%
net = ImageNeuralNetwork(64, 4, 5).to(device)
loss_function = nn.CrossEntropyLoss(label_smoothing = 0.1)
optimizer = optim.SGD(net.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 1e-4, nesterov = True)
scheduler = CosineAnnealingLR(optimizer, T_max = 300, eta_min = 1e-7)

# %%
for epoch in range(300):
    print(f'Training epoch {epoch}...')
    
    running_loss = 0.0
    
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.to(device) 
        labels = labels.to(device)

        optimizer.zero_grad()
        
        outputs = net(inputs)

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    print(f'Loss: {running_loss / len(train_loader):.4f}, LR: {current_lr:.6f}')

# %%
correct = 0
total = 0

net.eval()
with torch.no_grad(): 
    for data in test_loader:
        images, labels = data
        images = images.to(device) 
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy}%')

# %%
torch.save(net.state_dict(), f'trained_net_{accuracy}.pth')

# %%
net = ImageNeuralNetwork().to(device)
# net.load_state_dict(torch.load(f'trained_net_{accuracy}.pth'))
net.load_state_dict(torch.load('trained_net_91.01.pth'))

# %%
new_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# %%
def load_image(image_path):
    image = Image.open(image_path)
    image = new_transform(image)
    image = image.unsqueeze(0)
    return image

# %%
image_paths = ['animal.jpg']
images = [load_image(img) for img in image_paths]

# %%
net.eval()
with torch.no_grad():
    for image in images:
        outputs = net(image)
        _, predicted = torch.max(outputs, 1)
        print(f'Prediction: {class_names[predicted.item()]}')
