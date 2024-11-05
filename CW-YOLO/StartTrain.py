from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/CW-YOLO.yaml')
    # 使用yaml配置文件来创建模型,并导入预训练权重.
    model.train(**{'cfg': 'ultralytics/cfg/default.yaml', 'data': 'ultralytics/cfg/datasets/cottenweeds.yaml'})
   # model.train(**{'cfg': 'ultralytics/cfg/exp1.yaml', 'data': 'D:/yolov8/YoloV8Source/ultralytics/cfg/datasets/MyDataSets'})