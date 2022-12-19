# 目录

<!-- TOC -->

- [目录](#目录)
- [Twin描述](#Twin描述)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
- [训练和测试](#训练和测试)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [ImageNet-1k上的Twin训练](#ImageNet-1k上的Twin训练)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# [Twin描述](#目录)

PVT引入金字塔多级设计来很好的应对稠密估计任务，如目标检测和语义分割，但 PVT 比 Swin-Transformer 的效果略低，作者发现导致效果略低的原因应该就在于 PVT 的 ”绝对位置编码” 方法。因为分类任务和稠密预测任务有很大的不同，尤其在训练阶段，分类任务只需要处理输入大小固定的图片，而检测等任务的输入大小是不同的，提前设定好的绝对位置编码难以处理不同输入大小，而且会破坏平移不变性。
但 Swin-Transformer 使用了相对位置编码，会克服上述困难。
于是作者就使用了CPVT 中提出的 conditional position encoding (CPE)，克服了上述困难，在分类和其他任务上都取得了性能提升，可以媲美 Swin-Transformer，由于条件位置编码 CPE 支持输入可变长度，使得 Transformer 能够灵活处理不同尺度的输入。


# [数据集](#目录)

使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：共1000个类、224*224彩色图像
    - 训练集：共1,281,167张图像
    - 测试集：共50,000张图像
- 数据格式：JPEG
    - 注：数据在dataset.py中处理。
- 下载数据集，目录结构如下：

 ```text
└─imagenet
    ├─train                 # 训练数据集
    └─val                   # 评估数据集
 ```

# [特性](#目录)

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.9/others/mixed_precision.html?highlight=%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6)
的训练方法，使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。

# [环境要求](#目录)

- 硬件（Ascend）
    - 使用Ascend来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# [脚本说明](#目录)

## 脚本及样例代码

```text
└── Twin
    ├── eval.py									# 推理文件
    ├── export.py								# 权重导出文件
    ├── README_CN.md							# 说明文件
    ├── requriments.txt							# 
    ├── scripts
    │   ├── run_distribute_train_ascend.sh		# 卡Ascend910训练脚本
    │   ├── run_eval_ascend.sh					# Ascend测试脚本
    │   └── run_standalone_train_ascend.sh		# 单卡Ascend910训练脚本
    ├── src
    │   ├── args.py								# 参数文件
    │   ├── configs					
    │   │   ├── parser.py	
    │   │   ├── pth2ckpt				
    │   │   │   └── pth2ckpt.ipynb				# pytorch预训练权重转mindspore
    │   │   └── twin_s.yaml.yaml	            # twin_s.yaml.yaml的配置文件
    │   ├── data
    │   │   ├── augment							# 数据增强文件
    │   │   │   ├── auto_augment.py
    │   │   │   ├── __init__.py
    │   │   │   ├── mixup.py
    │   │   │   ├── random_erasing.py
    │   │   │   └── transforms.py
    │   │   ├── data_utils						# obs交互文件
    │   │   │   ├── __init__.py
    │   │   │   └── moxing_adapter.py
    │   │   ├── imagenet.py						# ImageNet数据类
    │   │   └── __init__.py
    │   ├── models								# 模型定义
    │   │   ├── __init__.py
    │   │   ├── layers
    │   │   │   ├── drop_path.py
    │   │   │   ├── attention.py
    │   │   │   └── identity.py
    │   │   └── twin.py
    │   ├── tools
    │   │   ├── callback.py						# 回调函数
    │   │   ├── cell.py							# 关于cell的自定义类
    │   │   ├── criterion.py					# 损失函数
    │   │   ├── get_misc.py						# 功能函数
    │   │   ├── __init__.py		
    │   │   ├── var_init.py                     # 参数初始化		
    │   │   ├── optimizer.py					# 优化器文件
    │   │   └── schedulers.py					# 学习率策略
    │   └── trainer
    │       └── train_one_step.py				# 自定义单步训练
    └── train.py								# 训练文件						

```

## 脚本参数

在twin_s.yaml.yaml中可以同时配置训练参数和评估参数。

- 配置Twin和ImageNet-1k数据集。

  ```text
    # Architecture 81.2%
    arch: pcpvt_small_v0
    
    # ===== Dataset ===== #
    data_url: ../data/imagenet											# 数据集地址
    set: ImageNet														# 数据集名称
    num_classes: 1000													# 分类数据的类别数
    mix_up: 0.8															# MixUp系数
    cutmix: 1.0															# CutMix系数
    auto_augment: rand-m9-mstd0.5-inc1									# 随机数据增强参数
    interpolation: bicubic												# 插值方法
    re_prob: 0.25														# random_erasing参数
    re_mode: pixel														# random_erasing参数
    re_count: 1															# random_erasing参数
    mixup_prob: 1.0														# mix_up的概率
    switch_prob: 0.5													# mixup和cutmix的转换概率
    mixup_mode: batch													# mixup的模式
    image_size: 224														# 图像大小
    crop_pct: 0.875														# 图像缩放比例
    
    # ===== Learning Rate Policy ======== #
    optimizer: adamw													# 优化器类别
    base_lr: 0.001													    # 基础学习率
    warmup_lr: 0.000001												    # 学习率热身初始值
    min_lr: 0.00001													    # 最小学习率
    lr_scheduler: cosine_lr												# 学习率变换策略
    warmup_length: 20													# 学习率热身轮数
    
    # ===== Network training config ===== #
    amp_level: O1														# 混合精度级别
    keep_bn_fp32: True													# 是否保持BN为FP32运算
    beta: [ 0.9, 0.999 ]												# 优化器一阶、二阶梯度系数
    clip_global_norm_value: 5.											# 全局梯度裁剪范数
    is_dynamic_loss_scale: True											# 是否为动态的损失缩放
    epochs: 300															# 训练轮数
    cooldown_epochs: 10													# 学习率冷却稀疏
    label_smoothing: 0.1												# 标签平滑稀疏
    weight_decay: 0.05											        # 权重衰减系数
    momentum: 0.9														# 优化器动量
    batch_size: 128														# 单卡批次大小
    drop_path_rate: 0.1													# drop path概率
    
    # ===== Hardware setup ===== #
    num_parallel_workers: 16											# 数据预处理线程数
    device_target: Ascend												# 设备选择
  ```

更多配置细节请参考脚本`twin_s.yaml`。 通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

# [训练和测试](#目录)

- Ascend处理器环境运行

  ```bash
  # 使用python启动单卡训练
  python train.py --device_id 0 --device_target Ascend --config ./src/configs/twin_s.yaml.yaml \
  > train.log 2>&1 &
  
  # 使用脚本启动单卡训练
  bash ./scripts/run_standalone_train_ascend.sh [DEVICE_ID] [CONFIG_PATH]
  
  # 使用脚本启动多卡训练
  bash ./scripts/run_distribute_train_ascend.sh [RANK_TABLE_FILE] [CONFIG_PATH]
  
  # 使用python启动单卡运行评估示例
  python eval.py --device_id 0 --device_target Ascend --config ./src/configs/twin_s.yaml.yaml > ./eval.log 2>&1 &
  
  # 使用脚本启动单卡运行评估示例
  bash ./scripts/run_eval_ascend.sh [DEVICE_ID] [CONFIG_PATH] [CHECKPOINT_PATH]
  ```
  
对于分布式训练，需要提前创建JSON格式的hccl配置文件。
  
请遵循以下链接中的说明：
  
[hccl工具](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)


# [模型描述](#目录)

## 性能

### 评估性能

#### ImageNet-1k上的Twin训练

| 参数                 | Ascend                          |
| -------------------------- |---------------------------------|
|模型| Twin                            |
| 模型版本              | pcpvt_small_v0                  |
| 资源                   | Ascend 910 8卡                   |
| 上传日期              | 2022-11-04                      |
| MindSpore版本          | 1.5.1                           |
| 数据集                    | ImageNet-1k Train，共1,281,167张图像 |
| 训练参数        | epoch=300, batch_size=1024      |
| 优化器                  | AdamWeightDecay                 |
| 损失函数              | SoftTargetCrossEntropy          |
| 损失| 0.627                           |
| 输出                    | 概率                              |
| 分类准确率             | 八卡：top1:81.35% top5:95.72%      |
| 速度                      | 8卡：371.446毫秒/步                  |
| 训练耗时          | 42h30min03s（run on OpenI）       |


# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)