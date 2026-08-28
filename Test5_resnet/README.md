# ResNet 花卉分类教程

## 原教程文件

- `model.py`：手写 ResNet-34/50/101 和 ResNeXt。
- `train.py`：读取 `ImageFolder` 数据、加载 ImageNet 权重、替换最后的全连接层并训练。
- `predict.py`：单张图片预测。
- `batch_predict.py`：批量图片预测。
- `load_weights.py`：演示加载预训练权重。

## 推荐的现代版本

- `model_modern.py`：只保留本例真正用到的 BasicBlock 和 ResNet-18/34，跳过暂时用不到的 Bottleneck/ResNeXt 参数。
- `train_modern.py`：支持两种数据布局，并在扁平花卉目录中自动做可复现的分层验证集划分。
- `predict_modern.py`：支持单图或目录预测，支持 jpg/jpeg/png/bmp/webp，不会漏掉最后一个不完整 batch。

项目中已有 `data/flower_photos.tgz`。先解压：

```powershell
tar -xzf data/flower_photos.tgz -C data
```

然后训练。花卉数据量较小，实际练习推荐迁移学习；权重会由 torchvision 下载一次并缓存：

```powershell
python Test5_resnet/train_modern.py --imagenet-weights --epochs 10
```

如果只想观察完整的从零训练流程，可以去掉 `--imagenet-weights`。

默认会读取 `data/flower_photos/<类别>/*.jpg`，并按类别抽取 20% 做验证；也可以使用传统的
`data_root/train/<类别>` 与 `data_root/val/<类别>` 目录，此时不会再次切分。结果保存在
`Test5_resnet/runs/best.pt`，其中包含模型参数、类别名、epoch 和验证准确率。

预测示例：

```powershell
python Test5_resnet/predict_modern.py --image path/to/flower.jpg
python Test5_resnet/predict_modern.py --image-dir path/to/images
```

如果已经下载了与本手写模型结构匹配的 ImageNet ResNet-34 `state_dict`，也可以改用本地文件：

```powershell
python Test5_resnet/train_modern.py --pretrained-path Test5_resnet/resnet34-pre.pth
```

## 代码怎样实现残差学习

一个 BasicBlock 的主分支是 `3x3 卷积 -> BN -> ReLU -> 3x3 卷积 -> BN`，旁路分支是恒等映射
`x`。最后计算 `ReLU(F(x) + x)`。当步幅改变或通道数改变时，旁路换成 `1x1 卷积 + BN`，把
`x` 投影到与 `F(x)` 相同的形状。ResNet-34 的四个 stage 含有 `[3, 4, 6, 3]` 个 block，空间
尺寸在 stage 2/3/4 的第一个 block 中减半，通道数依次为 64/128/256/512；最后用自适应全局
平均池化得到一个 512 维向量，再交给分类器。

## 原代码中需要注意的旧式或实际问题

1. `train.py` 把数据写死为 `../../data_set/flower_data`，与本项目的 `data/flower_photos.tgz`
   不匹配；路径还依赖启动脚本时的当前目录。
2. 原脚本强制要求本地 `resnet34-pre.pth`，仓库中没有该文件；新版默认可以从零训练，并提供
   `--pretrained-path` 作为可选项。
3. `batch_predict.py` 的 `range(len(images) // batch_size)` 会跳过最后不足一个 batch 的图片，且
   只识别 `.jpg`。
4. 原训练循环没有随机种子、学习率调度、完整 checkpoint（只存模型参数），也没有按样本数统计
   loss；新版补齐了这些内容，并使用 `AdamW`、`CosineAnnealingLR`、`zero_grad(set_to_none=True)`。
5. `super(BasicBlock, self).__init__()`、大量相互兼容的 `**kwargs`、手动 `torch.unsqueeze` 等仍然
   能运行，但属于较旧、可读性较差的写法；新版使用 `super().__init__()`、明确的构造参数和
   `torch.inference_mode()`。
6. 原模型的残差结构本身没有过时：本例调用 `resnet34`，使用的是标准 BasicBlock；文件里额外
   定义的 Bottleneck 则采用 torchvision 常见的 ResNet v1.5 步幅安排。过时的主要是工程脚本和
   路径处理，而不是核心网络思想。
