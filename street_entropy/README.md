README：

---

# 🧠 项目逻辑梳理：这两个脚本分别做什么？

---

## 📦 1. `entropy_analysis.py` 是 **图像特征提取模块**

### ✅ 它的任务：
> 给一张图，提取一整套描述图像“复杂度”的数值特征（用于后续分析或训练模型）

### 🚀 输入：
- 单张图像路径 `image_path`

### 📊 输出：
- 一个字典：包含这些数值特征：
  - **全局信息熵**：`全局熵`
  - **局部熵均值、标准差、最大值、分位数、中位数**
  - **LEDI**：局部熵密度指数（= 局部均值 / 全局熵）
  - **边缘密度**（Canny边缘像素所占比例）
- 还有两个图：
  - 灰度图（可选画图）
  - 局部熵热力图（可选保存图）

---

## ✅ 核心函数是：

```python
extract_entropy_features(image_path)
```

调用这个函数就能得到一张图的全部“视觉复杂度特征”。

你可以把它看作一个**图像转结构向量的黑箱函数**。

---

## 💼 2. `batch_entropy_with_heatmap.py` 是 **批量控制与输出模块**

### ✅ 它的任务：
> 自动批量读取一整个目录下的图像（original 和 deepth），对每张图调用 `extract_entropy_features()`，提取结果后：
- 存成一张 `CSV` 表格
- 自动保存局部熵热力图为 `.jpg` 文件

---

### 🚀 输入：
```
input/
├── original/
│   ├── 1/xxx.png
│   ├── 2/xxx.png
├── deepth/
│   ├── 1/xxx.png
```

### 📊 输出：
```
熵特征提取结果.csv
output_heatmaps/
├── original_1_xxx_heatmap.jpg
├── deepth_2_xxx_heatmap.jpg
```

---

## ✅ 它的逻辑流程图：

```
遍历 original & deepth 目录 →
    每张图：
        调用 extract_entropy_features →
            保存数值特征到表格
            保存熵图为 jpg
```

---

## 🌟 你现在这个系统可以干嘛？

- 💡 分析城市街景的复杂度、结构、秩序感
- ⚖️ 为街道等级评分模型提供基础特征
- 🔥 训练模型识别不同城市肌理
- 🎨 视觉热力图用于展示“城市信息密度”
- 📊 自动化批量图像视觉感知分析系统！

---

## ✅ 你现在可以继续拓展：

| 想法 | 模块 |
|------|------|
| 保存边缘图 | 在 `entropy_analysis.py` 中生成边缘图 |
| 提取绿视率 / 天空比例 | 加图像分割模块（可选用颜色阈值或深度模型） |
| 模型训练接口 | 加一个训练器，把提取的特征+标签训练出评分模型 |
| 多图对比 UI | 加 `streamlit` 或 `tkinter` 做可视化对比界面 |

---

要不要我现在就为你 **重写优化版的 `entropy_analysis.py`**，逻辑更清晰、结构更分明，并支持保存熵图与边缘图？我们可以统一一下架构，让你整个项目进入“战斗状态”。⚙️📊🔥
