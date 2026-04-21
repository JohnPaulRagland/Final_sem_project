import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2
import torch.nn as nn

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Kidney Disease Detection", layout="wide")

# ---------- CLASS LABELS ----------
class_names = ['Cyst','Normal','Stone','Tumor']

# ---------- SIMPLE MEDICAL INFO ----------
condition_info = {
    "Cyst": {
        "simple": "There may be a fluid-filled sac in the kidney.",
        "action": "Usually not serious, but it is good to consult a doctor for confirmation."
    },
    "Normal": {
        "simple": "The kidney looks healthy and normal.",
        "action": "No issues detected. Maintain a healthy lifestyle."
    },
    "Stone": {
        "simple": "There may be a kidney stone present.",
        "action": "Drink more water and consult a doctor if pain occurs."
    },
    "Tumor": {
        "simple": "There may be abnormal growth in the kidney.",
        "action": "Please consult a doctor immediately for further testing."
    }
}

# ---------- LOAD MODEL ----------
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("ckd_model.pth", map_location="cpu"))
model.eval()

# ---------- TRANSFORM ----------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ---------- GRAD-CAM ----------
target_layer = model.layer4[1].conv2

def gradcam(image_tensor):
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

    grads = gradients[0].detach().numpy()[0]
    acts = activations[0].detach().numpy()[0]

    weights = np.mean(grads, axis=(1,2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224,224))
    cam = cam / cam.max()

    fh.remove()
    bh.remove()

    return cam, pred_class

# ---------- UI ----------
st.title("🧠 Kidney Disease Detection System")

st.write("Upload a CT scan image to detect kidney condition using AI.")

uploaded_file = st.file_uploader("Upload CT Scan Image", type=["jpg","png","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    # ---------- ORIGINAL IMAGE ----------
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    # ---------- PROCESS ----------
    image_tensor = transform(image).unsqueeze(0)

    cam, pred = gradcam(image_tensor)
    prediction = class_names[pred]

    # ---------- CONFIDENCE ----------
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        confidence = probs[0][pred].item() * 100

    # ---------- RESULT ----------
    with col2:
        st.subheader("Prediction Result")
        st.success(f"{prediction}")
        st.write(f"Confidence: {confidence:.2f}%")

    # ---------- SIMPLE EXPLANATION ----------
    info = condition_info[prediction]

    st.subheader("🩺 Simple Explanation")

    st.write("**What this means:**")
    st.success(info["simple"])

    st.write("**What you should do:**")
    st.warning(info["action"])

    # ---------- GRAD-CAM VISUALIZATION ----------
    image_np = np.array(image.resize((224,224))) / 255

    colored_cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    colored_cam = colored_cam / 255

    overlay = 0.5 * colored_cam + image_np
    overlay = np.clip(overlay, 0, 1)

    st.subheader("🔥 Explainable AI (Grad-CAM)")

    col3, col4, col5 = st.columns(3)

    with col3:
        st.image(image_np, caption="Original")

    with col4:
        st.image(colored_cam, caption="Heatmap")

    with col5:
        st.image(overlay, caption="Overlay")

    # ---------- GRAD-CAM EXPLANATION ----------
    st.subheader("🔍 Understanding the Highlighted Image")

    st.write("""
- The colored areas show where the AI is focusing in the scan.
- Red / Yellow areas → Most important regions.
- Blue areas → Less important regions.

👉 In simple words:  
The model is looking at the highlighted area to decide the result.
""")

    st.info("The highlighted region shows the part of the kidney that influenced the AI prediction the most.")

# ---------- DISCLAIMER ----------
st.markdown("---")
st.caption("⚠️ This system is for educational purposes only and should not replace professional medical advice.")