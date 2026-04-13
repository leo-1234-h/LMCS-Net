import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
  model = YOLO('ultralytics/cfg/models/v10/yolov10_LMCS-Net.yaml')
  model.load('yolov10n.pt')
  results = model.train(
    data='data.yaml',  #数据集配置文件的路径
    epochs=200,  #训练轮次总数
    batch=8,  #批量大小，即单次输入多少图片训练
    imgsz=640,  #训练图像尺寸
    workers=8,  #加载数据的工作线程数
    device= '0',  #指定训练的计算设备，无nvidia显卡则改为 'cpu'
    optimizer='AdamW',  #训练使用优化器，可选 auto,SGD,Adam,AdamW 等
)