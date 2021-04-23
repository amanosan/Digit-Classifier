import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.output_layer(out)
        return out


input_size = 784
hidden_size = 500
num_classes = 10
device = torch.device('cpu')
model = NeuralNet(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load(
    "./app/digit_classifier.pth", map_location=device))
model.eval()


def transform_image(image_bytes):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1302,), (0.3081, ))
        ]
    )

    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)


def get_prediction(image_tensor):
    image = image_tensor.reshape(-1, 28*28)
    outputs = model(image)

    _, predicted = torch.max(outputs.data, 1)

    return predicted
