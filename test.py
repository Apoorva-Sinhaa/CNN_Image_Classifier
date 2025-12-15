import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import CNNClassifier

classes = (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


model = CNNClassifier().to(device)
model.load_state_dict(torch.load("cifar10_cnn.pth", map_location=device))
model.eval()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=True
)


with torch.no_grad():
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    plt.figure(figsize=(14, 4))
    for i in range(8):
        img = images[i].cpu() * 0.5 + 0.5  
        plt.subplot(1, 8, i + 1)
        plt.imshow(img.permute(1, 2, 0))
        color = "green" if predicted[i] == labels[i] else "red"
        plt.title(
            f"T:{classes[labels[i]]}\nP:{classes[predicted[i]]}",
            color=color
        )
        plt.axis("off")

    plt.suptitle("CIFAR-10 CNN Predictions")
    plt.show()


correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
