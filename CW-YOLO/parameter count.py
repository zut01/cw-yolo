import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # choose your yaml file
    model = YOLO('ultralytics/cfg/models/yolov8.yaml',task='detect')
    model.info(detailed=True)
    model.profile(imgsz=[640 ])
    model.fuse()
  #  print(d["head"])