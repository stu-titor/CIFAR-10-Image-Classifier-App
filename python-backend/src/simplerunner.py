from PIL import Image
import torch
import torchvision.transforms as transforms
from CNN import ImageNeuralNetwork

def load_image(image_path):
    image = Image.open(image_path)
    image = new_transform(image)
    image = image.unsqueeze(0)
    return image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Use the same parameters that were used to create the model
net = ImageNeuralNetwork(64, 4, 5)

net.load_state_dict(torch.load("trained_net_91.01.pth", map_location = device, weights_only = False))

new_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# Classify all images in image_paths
image_paths = ['animal.jpg']
images = [load_image(img) for img in image_paths]
net.eval()
with torch.no_grad():
    for image in images:
        outputs = net(image)
        _, predicted = torch.max(outputs, 1)
        print(f'Prediction: {class_names[predicted.item()]}')
