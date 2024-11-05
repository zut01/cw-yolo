from ultralytics import YOLO
model = YOLO('best.pt')
# model.predict(source='datasets/data/wireworm 0.8', save=True)
results = model("D:/yolov8/vis/24.jpg", visualize=True)