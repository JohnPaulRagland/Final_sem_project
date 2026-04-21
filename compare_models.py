import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from PIL import Image

# ---------- DATASET ----------
dataset_path = "dataset"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

dataset = datasets.ImageFolder(dataset_path, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

class_names = dataset.classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- MODELS ----------
def get_models(num_classes):
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, num_classes)

    mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    mobilenet.classifier[1] = nn.Linear(mobilenet.classifier[1].in_features, num_classes)

    return {"ResNet18": resnet, "VGG16": vgg, "MobileNet": mobilenet}

# ---------- TRAIN FUNCTION ----------
def train_model(model):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(3):  # keep small for speed
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model

# ---------- EVALUATION ----------
def evaluate_model(model):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return acc, prec, rec, f1, y_true, y_pred

# ---------- GRAD-CAM ----------
def gradcam(model, image_tensor, target_layer):
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    pred_class = output.argmax()

    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0].detach().cpu().numpy()[0]
    acts = activations[0].detach().cpu().numpy()[0]

    weights = np.mean(grads, axis=(1,2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224,224))
    cam = cam / cam.max()

    fh.remove()
    bh.remove()

    return cam

# ---------- MAIN ----------
models_dict = get_models(len(class_names))
results = {}

for name, model in models_dict.items():
    print(f"\nTraining {name}...")
    model = train_model(model)

    acc, prec, rec, f1, y_true, y_pred = evaluate_model(model)
    results[name] = acc

    print(f"{name} Accuracy: {acc:.4f}")

# ---------- COMPARISON GRAPH ----------
plt.bar(results.keys(), results.values())
plt.title("Model Comparison (Accuracy)")
plt.ylabel("Accuracy")
plt.show()

# ---------- CONFUSION MATRIX FOR BEST MODEL ----------
best_model_name = max(results, key=results.get)
print("\nBest Model:", best_model_name)

best_model = models_dict[best_model_name].to(device)
_, _, _, _, y_true, y_pred = evaluate_model(best_model)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title(best_model_name + " Confusion Matrix")
plt.show()

# ---------- GRAD-CAM DEMO ----------
sample_img_path = dataset.samples[0][0]
img = Image.open(sample_img_path).convert("RGB")

input_tensor = transform(img).unsqueeze(0).to(device)

# pick layer based on model
if best_model_name == "ResNet18":
    target_layer = best_model.layer4[1].conv2
elif best_model_name == "VGG16":
    target_layer = best_model.features[-1]
else:
    target_layer = best_model.features[-1]

cam = gradcam(best_model, input_tensor, target_layer)

img_np = np.array(img.resize((224,224))) / 255
heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)/255
overlay = 0.5*heatmap + img_np

plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.imshow(img_np)
plt.title("Original")

plt.subplot(1,3,2)
plt.imshow(cam, cmap='jet')
plt.title("Grad-CAM")

plt.subplot(1,3,3)
plt.imshow(overlay)
plt.title("Overlay")

plt.show()