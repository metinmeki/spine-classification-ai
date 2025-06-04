from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import torch
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import uuid
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn.functional as F

# ========== CONFIG ==========
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}

# ========== FLASK SETUP ==========
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ========== LOAD CUSTOM MODEL ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomResNet50, self).__init__()
        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.base_model.fc = nn.Identity()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        if x.ndim == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x

clf_model = CustomResNet50(num_classes=2).to(device)
clf_model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
clf_model.eval()

# ========== GRAD-CAM HOOKS ==========
gradients = None
activations = None

def save_gradients_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

def save_activations(out):
    global activations
    activations = out

def get_heatmap(input_tensor):
    global gradients, activations

    model = clf_model.base_model
    target_layer = model.layer4[-1].conv2
    target_layer.register_forward_hook(lambda m, inp, out: save_activations(out))
    target_layer.register_backward_hook(save_gradients_hook)

    clf_model.zero_grad()
    output = clf_model(input_tensor)
    pred_class = torch.argmax(output, dim=1)
    class_score = output[0, pred_class]
    class_score.backward()

    pooled_grad = torch.mean(gradients, dim=[0, 2, 3])
    heatmap = activations[0].detach().cpu().numpy()

    for i in range(len(pooled_grad)):
        heatmap[i, :, :] *= pooled_grad[i].item()

    heatmap = np.mean(heatmap, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

# ========== UTILS ==========
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_dicom_image(path):
    dicom = pydicom.dcmread(path)
    img = apply_voi_lut(dicom.pixel_array, dicom)
    img = img.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0
    return Image.fromarray(img).convert("RGB")

def preprocess_image(image_path):
    if image_path.lower().endswith(".dcm"):
        image = read_dicom_image(image_path)
    else:
        image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image).unsqueeze(0).to(device), image.resize((256, 256))

def classify_image_with_confidence(pil_image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = clf_model(input_tensor)
        probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probabilities))
        confidence = float(probabilities[pred])
    label = "Disorder" if pred == 1 else "Normal"
    return label, confidence, input_tensor

def save_result_image(image_pil, label, result_path, input_tensor):
    heatmap = get_heatmap(input_tensor)
    heatmap = Image.fromarray(np.uint8(255 * heatmap)).resize(image_pil.size)
    heatmap = np.array(heatmap)
    heatmap_colored = plt.cm.jet(heatmap)[:, :, :3] * 255
    heatmap_colored = heatmap_colored.astype(np.uint8)
    overlay = Image.blend(image_pil.convert("RGBA"), Image.fromarray(heatmap_colored).convert("RGBA"), alpha=0.5)
    overlay.save(result_path)

# ========== ROUTES ==========
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            image_tensor, image_pil = preprocess_image(filepath)
            label, confidence, input_tensor = classify_image_with_confidence(image_pil)

            result_id = str(uuid.uuid4())
            result_path = os.path.join(RESULT_FOLDER, f"result_{result_id}.png")
            original_result_path = os.path.join(RESULT_FOLDER, f"original_{result_id}.png")

            save_result_image(image_pil, label, result_path, input_tensor)
            image_pil.save(original_result_path)

            return render_template('index.html',
                                   result_img=result_path.replace("\\", "/"),
                                   original_img=original_result_path.replace("\\", "/"),
                                   label=label,
                                   confidence=confidence)

    return render_template('index.html', result_img=None, original_img=None)

# ========== MAIN ==========
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    app.run(debug=True)
