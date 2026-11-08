# DFHMG-Net: 融合双频膨胀与混合流形图的三维牙齿分割
**3D Tooth Segmentation Fusing Dual-Frequency Dilation and Hybrid Manifold Graph**

本仓库为论文 **《融合双频膨胀与混合流形图的三维牙齿分割》** 的官方 PyTorch 源代码实现。

---

## 1. 简介 (Introduction)
针对三维口内扫描牙齿分割中因局部感受野受限导致的几何拓扑感知不足，以及牙齿交界处缺乏高低频特征解耦而引发的边界粘连与过分割问题，本文提出了一种轻量级三维牙齿分割网络 **DFHMG-Net**（参数量仅 **1.32M**）。

### 核心模块：
* **双频膨胀图卷积模块 (DFD-GCN)**：采用空间内外环隔离策略，内环提取拉普拉斯高频差分响应以锐化牙缝边缘，外环结合膨胀最远点采样（FPS）捕获大感受野低频拓扑上下文，并通过门控机制自适应融合。
* **空间锚定混合流形图 (SAHM)**：将归一化三维物理坐标与高维语义特征联合映射至混合流形空间动态构建拓扑连接，抑制深层特征同质化引起的边界漂移。
* **双域注意力机制 (DDA)**：协同通道级统计池化与空间级局部邻域差分响应，实现多尺度特征的精准重标定。

---

## 2. 项目结构 (Project Structure)
```text
DFHMG-Net/
├── dataset/                # 数据集加载与预处理定义
├── images/                 # 模型架构与示例图片
├── models/                 # DFHMG-Net 核心网络与模块实现
├── pointnet2_ops_lib/      # 点云操作算子库 (FPS / k-NN 等)
├── utils/                  # 评价指标与辅助工具函数
├── Dockerfile              # Docker 容器环境配置
├── project_structure.txt   # 目录结构说明
├── requirements.txt        # Python 依赖包列表
├── train.py                # 模型训练脚本
├── test.py                 # 模型测试与评估脚本
└── visualize_example.py    # 三维牙颌分割结果可视化脚本
