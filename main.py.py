import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import gradio as gr
from collections import OrderedDict
import os
import urllib.request

# Number of classes (2 lesion classes + background)
num_classes = 3
class_names = ["Benign", "Malignant"]

# Download model if not exists (replace URL with your own model link)
model_path = "fasterrcnn_model.pth"
if not os.path.exists(model_path):
    print("Downloading model...")
    urllib.request.urlretrieve(
        "https://your-cloud-link.com/fasterrcnn_model.pth", model_path
    )

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

    for box, label, score in zip(
        outputs["boxes"], outputs["labels"], outputs["scores"]
    ):
        if score > 0.5 and label > 0:
            x1, y1, x2, y2 = box.tolist()
            draw_box.rectangle([x1, y1, x2, y2], outline="red", width=2)
            text = f"{class_names[label - 1]}: {score:.2f}"
            draw_box.text((x1, y1 - 10), text, fill="red")

    return draw


demo = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Faster R-CNN Breast Lesion Detection",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
