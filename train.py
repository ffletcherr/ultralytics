from ultralytics.models.yolo.detect import DetectionTrainer


trainer = DetectionTrainer(
    overrides={
        "model": "yolov8n",
        "epochs": 1,
        "data": "./ultralytics/cfg/datasets/coco8.yaml",
        "augment": False,
    }
)
trainer.train()
