import joblib
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch 
import torch.nn as nn
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_filename = "FER-CNN.pkl"
model = joblib.load(model_filename)
model.to(device)
model.eval()
print("✅ Model loaded successfully!")
dataset_root = "./data"
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.ImageFolder(root=f"{dataset_root}/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"📂 Loaded {len(test_dataset)} test images.")

correct = 0
total = 0
criterion = nn.CrossEntropyLoss()

with torch.no_grad():  # No need to compute gradients during testing
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # Get class with highest probability
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"✅ Model Accuracy on Test Data: {accuracy:.2f}%")