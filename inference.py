from ultralytics.models import YOLO
from torch import nn
import torch

model = YOLO("yolov8n.yaml")
class_names = model.names
image_path = "./ultralytics/assets/bus.jpg"
for idx in range(3):
    print("idx:", idx)
    results = model([image_path, image_path])
