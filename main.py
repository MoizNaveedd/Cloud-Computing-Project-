import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Attack Libraries
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from torchattacks import FGSM

# Data type for info display
DATA_TYPE = "image"

# Define CNN Model
class ResilienceCNN(nn.Module):
    def __init__(self, num_classes=10):  # ✅ fixed constructor name
        super(ResilienceCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load Dataset
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])

train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# Initialize model, loss, optimizer
model = ResilienceCNN(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
fgsm = FGSM(model, eps=0.1)

print("Training model with adversarial examples...")

# # Train Model
# for epoch in range(5):
#     for images, labels in train_loader:
#         images, labels = images, labels

#         # Generate adversarial examples
#         adv_images = fgsm(images, labels)

#         # Train on both clean and adversarial images
#         optimizer.zero_grad()
#         loss = criterion(model(images), labels) + criterion(model(adv_images), labels)
#         loss.backward()
#         optimizer.step()

#     print(f"Epoch {epoch+1} completed.")

# Save Model
# torch.save(model.state_dict(), "resilience_model.pth")
# print("Model saved!")

# Load Model
model.load_state_dict(torch.load("resilience_model.pth"))
model.eval()
print("✅ Model loaded successfully!")

# Wrap Model for ART Adversarial Testing
art_classifier = PyTorchClassifier(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 32, 32),
    nb_classes=10,
)

# Use ART's FastGradientMethod to generate adversarial examples
attack = FastGradientMethod(estimator=art_classifier, eps=0.1)

# Get a batch of test images
test_images, test_labels = next(iter(test_loader))
test_images_np = test_images.numpy()

# Generate adversarial examples on this batch
adv_images = attack.generate(x=test_images_np)

# Evaluate Clean Accuracy
clean_preds = np.argmax(art_classifier.predict(test_images_np), axis=1)
clean_accuracy = np.mean(clean_preds == test_labels.numpy()) * 100

# Evaluate Adversarial Accuracy
adv_preds = np.argmax(art_classifier.predict(adv_images), axis=1)
adv_accuracy = np.mean(adv_preds == test_labels.numpy()) * 100

# Robustness Score
robustness_score = np.mean(clean_preds == adv_preds) * 100

# Print Results
print(f"✅ Clean Accuracy: {clean_accuracy:.2f}%")
print(f"🔴 Adversarial Accuracy: {adv_accuracy:.2f}%")
print(f"🛡️ Robustness Score: {robustness_score:.2f}%")

# Visualize the Results
metrics = ['Clean Accuracy', 'Adversarial Accuracy', 'Robustness']
scores = [clean_accuracy, adv_accuracy, robustness_score]

plt.bar(metrics, scores, color=['green', 'red', 'blue'])
plt.title("AI Resilience Performance")
plt.ylim(0, 100)
plt.ylabel("Accuracy (%)")
plt.show()

# Custom Image Test (Local)
def test_custom_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return

    custom_image = Image.open(image_path).convert('RGB').resize((32, 32))
    custom_image_tensor = transform(custom_image).unsqueeze(0)

    # Get model's prediction on the clean image
    clean_output = model(custom_image_tensor)
    clean_prediction = torch.argmax(clean_output, dim=1).item()

    # Generate an adversarial example using ART
    custom_image_np = custom_image_tensor.numpy()
    adv_custom_image = attack.generate(x=custom_image_np)
    adv_output = model(torch.from_numpy(adv_custom_image).float())
    adv_prediction = torch.argmax(adv_output, dim=1).item()

    # Display clean image
    plt.imshow(custom_image)
    plt.title(f"Clean Image Prediction: {clean_prediction}")
    plt.axis('off')
    plt.show()

    # Display adversarial image
    adv_custom_image_pil = Image.fromarray((adv_custom_image[0].transpose(1, 2, 0) * 255).astype(np.uint8))
    plt.imshow(adv_custom_image_pil)
    plt.title(f"Adversarial Image Prediction: {adv_prediction}")
    plt.axis('off')
    plt.show()

    # Safe or not
    if clean_prediction == adv_prediction:
        print("✅ The image is SAFE under adversarial attack.")
    else:
        print("❌ The image is NOT SAFE; adversarial attack changed the prediction.")

# Example Usage:
image_path = input("Enter the full path to your image file: ")
test_custom_image(image_path)
