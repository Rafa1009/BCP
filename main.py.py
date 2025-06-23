import os
import urllib.request
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
from collections import OrderedDict

# Model download setup
model_path = "fasterrcnn_model.pth"
if not os.path.exists(model_path):
    print("Downloading model...")
    url = "https://drive.google.com/uc?export=download&id=1x-3968oXNe1G1peobX04_Pn7OHbFIj1-"
    urllib.request.urlretrieve(url, model_path)

# Number of classes (2 classes + background)
num_classes = 3
class_names = ["Benign", "Malignant"]

def load_model():
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    checkpoint = torch.load(model_path, map_location="cpu")

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model

model = load_model()

def detect(image):
    image_tensor = F.to_tensor(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)[0]

    draw = image.copy()
    draw_box = ImageDraw.Draw(draw)

    # Optional: load a font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
        if score > 0.5 and label > 0:
            x1, y1, x2, y2 = box.tolist()
            draw_box.rectangle([x1, y1, x2, y2], outline="red", width=2)
            text = f"{class_names[label - 1]}: {score:.2f}"
            draw_box.text((x1, y1 - 20), text, fill="red", font=font)

    return draw

demo = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Faster R-CNN Breast Lesion Detection",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)

