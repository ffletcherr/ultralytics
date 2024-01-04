from ultralytics.models.yolo.detect import DetectionTrainer


trainer = DetectionTrainer(
    overrides={"model": "yolov8n", "data": "./ultralytics/cfg/datasets/coco8.yaml"}
)
