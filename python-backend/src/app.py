from flask import Flask, request, jsonify
import requests
import io

from PIL import Image
import torch
import torchvision.transforms as transforms
from CNN import ImageNeuralNetwork

app = Flask(__name__)

def load_image(image_file):
    image = Image.open(image_file)
    image = new_transform(image)
    image = image.unsqueeze(0)
    return image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Use the same parameters that were used to create the model
net = ImageNeuralNetwork(64, 4, 5)
net.load_state_dict(torch.load("trained_net_91.01.pth", map_location = device, weights_only = False))
net.eval()

new_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

@app.route('/classify/file', methods=['POST'])
def fileClassify():
    file = request.files['image']
    loaded_image = load_image(file)

    with torch.no_grad():
        _, predicted = torch.max(net(loaded_image), 1)
    return jsonify({"prediction": class_names[predicted.item()]})

@app.route('/classify/url', methods=['POST'])
def urlClassify():
    url = request.get_json()['url']
    file = io.BytesIO(requests.get(url).content)
    loaded_image = load_image(file)

    with torch.no_grad():
        _, predicted = torch.max(net(loaded_image), 1)
    return jsonify({"prediction": class_names[predicted.item()]})

if __name__ == '__main__':
    app.run(port=5000)
