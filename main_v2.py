import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
# Set matplotlib to use Agg backend (non-interactive, no GUI needed)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

# Attack Libraries
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from torchattacks import FGSM

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

# Define CNN Model
class ResilienceCNN(nn.Module):
    def __init__(self, num_classes=10):
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

# Define transforms
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])

# CIFAR10 class names for better output
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def evaluate_model():
    """Evaluate model performance and save results as image"""
    print("Loading test data...")
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    
    print("Loading model...")
    model = ResilienceCNN(num_classes=10)
    model.load_state_dict(torch.load("resilience_model.pth"))
    model.eval()
    print("✅ Model loaded successfully!")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
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
    
    # Visualize the Results and save to file
    metrics = ['Clean Accuracy', 'Adversarial Accuracy', 'Robustness']
    scores = [clean_accuracy, adv_accuracy, robustness_score]
    
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, scores, color=['green', 'red', 'blue'])
    plt.title("AI Resilience Performance")
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.savefig('results/performance_metrics.png')
    plt.close()
    
    print("Performance metrics saved to results/performance_metrics.png")
    
    return model, attack

def test_custom_image(image_path, output_path='results/comparison.png'):
    """Test a custom image and save comparison to file"""
    print(f"Processing image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return None
    
    # Load model
    model = ResilienceCNN(num_classes=10)
    model.load_state_dict(torch.load("resilience_model.pth"))
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Wrap Model for ART testing
    art_classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
    )
    
    # Create attack
    attack = FastGradientMethod(estimator=art_classifier, eps=0.1)
    
    # Load and preprocess image
    custom_image = Image.open(image_path).convert('RGB').resize((32, 32))
    custom_image_tensor = transform(custom_image).unsqueeze(0)
    
    # Get model's prediction on the clean image
    clean_output = model(custom_image_tensor)
    clean_prediction = torch.argmax(clean_output, dim=1).item()
    clean_class = class_names[clean_prediction]
    
    # Generate an adversarial example using ART
    custom_image_np = custom_image_tensor.numpy()
    adv_custom_image = attack.generate(x=custom_image_np)
    adv_output = model(torch.from_numpy(adv_custom_image).float())
    adv_prediction = torch.argmax(adv_output, dim=1).item()
    adv_class = class_names[adv_prediction]
    
    # Instead of showing, save comparison image to file
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(custom_image)
    plt.title(f"Clean: {clean_class} ({clean_prediction})")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    adv_custom_image_pil = Image.fromarray((adv_custom_image[0].transpose(1, 2, 0) * 255).astype(np.uint8))
    plt.imshow(adv_custom_image_pil)
    plt.title(f"Adversarial: {adv_class} ({adv_prediction})")
    plt.axis('off')
    
    # Save the comparison figure
    plt.savefig(output_path)
    plt.close()
    print(f"Comparison image saved to {output_path}")
    
    # Safe or not
    is_safe = clean_prediction == adv_prediction
    if is_safe:
        safety_message = "✅ The image is SAFE under adversarial attack."
    else:
        safety_message = "❌ The image is NOT SAFE; adversarial attack changed the prediction."
    
    print(safety_message)
    
    # Return results as dictionary
    return {
        'clean_prediction': clean_prediction,
        'clean_class': clean_class,
        'adv_prediction': adv_prediction,
        'adv_class': adv_class,
        'is_safe': is_safe,
        'safety_message': safety_message,
        'comparison_path': output_path
    }

# Command line interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python adversarial_ml.py evaluate")
        print("  python adversarial_ml.py test_image <image_path>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "evaluate":
        evaluate_model()
    
    elif command == "test_image" and len(sys.argv) >= 3:
        image_path = sys.argv[2]
        test_custom_image(image_path)
    
    else:
        print("Unknown command. Use 'evaluate' or 'test_image <image_path>'")