from mmdet.models import SparseRCNN
from mmdet.configs import get_config

cfg = get_config('/data1/lsl/lxw/mmdetection-main/configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py')

model = SparseRCNN(**cfg.model)

print("模型初始化成功")