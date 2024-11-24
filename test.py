from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the image
image_path = (r'C:\\Users\Owner\.cache\kagglehub\datasets\rajsahu2004\lacuna-malaria-detection-dataset\versions\1'
              r'/images/id_u3q6jdck4j.jpg')
image = Image.open(image_path).convert("RGB")

input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)


model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 3)

model.load_state_dict(torch.load("malaria_detection_model.pth", weights_only=True))
model.eval()

with torch.no_grad():
    outputs = model(input_batch)
    _, predicted = torch.max(outputs, 1)

class_labels = ["NEG", "Trophozoite", "WBC"]
predicted_class = class_labels[predicted.item()]

print(f"Predicted Class: {predicted_class}")