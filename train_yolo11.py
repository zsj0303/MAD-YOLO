# 在训练脚本开头添加
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
  # model = YOLO('runs/detect/train90/weights/best.pt')
  model = YOLO('ultralytics/cfg_yolo11/YOLO11-Add/yolov11n-ADown2-C2PSAMLCA-P2-DyHead.yaml',task='detect')
  model.load('yolo11n.pt')  # 注释则不加载

  # model.info(verbose=True)
  results = model.train(
    data='RS-STOD.yaml',  #数据集配置文件的路径
    resume=True,
    pretrained=True,
    epochs=200,  #训练轮次总数
    batch=4,  #批量大小，即单次输入多少图片训练
    imgsz=640,  #训练图像尺寸
    workers=2,  #加载数据的工作线程数
    device=0,  #指定训练的计算设备，无nvidia显卡则改为 'cpu'
    optimizer='AdamW',  #训练使用优化器，可选 auto,SGD,Adam,AdamW 等
    lr0=0.001,
    lrf = 0.01,
    # close_mosaic=10,
    warmup_epochs=5,# 延长预热
    box=7.0,  # 增加定位损失权重
    cls=0.5,  # 提高分类损失权重
    dfl=1.5,  # 加强分布聚焦损失
    label_smoothing=0.2,# 缓解类别不平衡
    weight_decay=0.0005,  # SGD使用较小的权重衰减
    augment=True,# AdamW建议使用更高的权重衰减
    mixup=0.0,
    copy_paste=0.0,  # 增强小目标
    plots=True,# 生成训练曲线
    val=True,
    amp=True,  #True 或者 False, 解释为：自动混合精度(AMP) 训练
    cos_lr=True,  # 余弦学习率调度
    mosaic=1.0,
    patience=0,  # 早停机制
    cache=False , # True 在内存中缓存数据集图像，服务器推荐开启
)

#
# # 在训练脚本开头添加
# import matplotlib
# matplotlib.use('Agg')  # 使用非交互式后端
#
# import warnings
# warnings.filterwarnings('ignore')
#
# from ultralytics import YOLO
#
#
# if __name__ == '__main__':
#
#     # 加载自定义模型结构
#     model = YOLO(
#         'ultralytics/cfg_yolo11/YOLO11/yolo11s.yaml',
#         task='detect'
#     )
#
#     # 加载 YOLO11n 预训练权重
#     model.load('yolo11s.pt')
#
#     results = model.train(
#         data='RS-STOD.yaml',
#
#         # ---------- 基础训练设置 ----------
#         resume=False,
#         pretrained=True,
#         epochs=200,
#         batch=4,
#         imgsz=640,
#         workers=2,
#         device=0,
#         cache=False,
#
#         # ---------- 优化器与学习率 ----------
#         optimizer='AdamW',
#         lr0=0.001,
#         lrf=0.01,
#         weight_decay=0.0005,
#         warmup_epochs=5,
#         cos_lr=True,
#
#         # ---------- 小目标检测损失权重 ----------
#         box=8.0,
#         cls=0.8,
#         dfl=1.5,
#
#         # TinyPerson 只有两类，不建议使用过强标签平滑
#         label_smoothing=0.0,
#
#         # ---------- 数据增强 ----------
#         augment=True,
#
#         # TinyPerson 小目标不建议过强 mosaic
#         mosaic=0.3,
#         close_mosaic=20,
#
#         # 小目标数据集不建议过强 mixup / copy_paste
#         mixup=0.05,
#         copy_paste=0.1,
#
#         # 基础几何增强
#         degrees=0.0,
#         translate=0.1,
#         scale=0.4,
#         shear=0.0,
#         perspective=0.0,
#         flipud=0.0,
#         fliplr=0.5,
#
#         # 颜色增强适当减弱
#         hsv_h=0.01,
#         hsv_s=0.5,
#         hsv_v=0.3,
#
#         # 不建议开启随机擦除，容易破坏小目标
#         erasing=0.0,
#
#         # ---------- 验证与保存 ----------
#         val=True,
#         plots=True,
#         amp=True,
#
#         # patience=0 表示不早停，适合论文实验完整训练
#         patience=0,
#
#         # 密集小目标建议提高最大检测数量
#         max_det=1000,
#     )