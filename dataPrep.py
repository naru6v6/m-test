import pandas as pd
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

import os

# Load the dataset
data = pd.read_csv('transformData.csv')

class_to_idx = {'NEG': 0, 'Trophozoite': 1, 'WBC': 2}
data['class'] = data['class'].map(class_to_idx)


# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['Path']
        image = Image.open(img_path).convert("RGB")
        label = self.dataframe.iloc[idx]['class']

        if self.transform:
            image = self.transform(image)

            # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)  # Ensure label is a LongTensor for classification

        return image, label


# Standardization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create Dataset and DataLoader
dataset = CustomDataset(data, transform=transform)
test_data = CustomDataset(data)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Model setup
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # or use ResNet18_Weights.DEFAULT for the latest weights
model.fc = torch.nn.Linear(model.fc.in_features, 3)  # Update output layer for 3 classes

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5

# Training loop (simplified)
for epoch in range(num_epochs):
    x = 0
    print(epoch)
    for inputs, labels in dataloader:
        x = x + 1
        print(x)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)  # Get the index of the max log-probability
    correct = (preds == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy


model_path = "malaria_detection_model.pth"

# Save the model's state_dict (parameters only)
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

model.eval()
test_accuracy = 0
num_samples = 0

with torch.no_grad():
    for inputs, labels in test_dataloader:  #
        outputs = model(inputs)
        accuracy = calculate_accuracy(outputs, labels)

        test_accuracy += accuracy * labels.size(0)
        num_samples += labels.size(0)

test_accuracy /= num_samples
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
