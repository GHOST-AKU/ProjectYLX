import numpy as np
from PIL import Image
from scipy.stats import entropy
from skimage.color import rgb2gray
from skimage.filters.rank import entropy as local_entropy
from skimage.morphology import disk
import cv2  # 用于边缘检测
import matplotlib.pyplot as plt

# ----------------------------
# 将彩色图像转为标准化灰度图（值在0~1之间）
# ----------------------------
def preprocess_image(img):
    gray = rgb2gray(np.array(img))      # 转为灰度图
    return gray / np.max(gray)          # 标准化为0~1之间

# ----------------------------
# 计算整张图的全局信息熵（香农熵）
# ----------------------------
def compute_global_entropy(gray_img, bins=256):
    hist, _ = np.histogram(gray_img, bins=bins, range=(0, 1), density=True)  # 灰度直方图
    hist = hist[hist > 0]  # 避免log(0)
    return entropy(hist, base=2)  # 香农熵计算

# ----------------------------
# 计算局部熵热力图（滑动窗口局部复杂度）
# ----------------------------
def compute_local_entropy_map(gray_img, radius=5):
    gray_uint8 = (gray_img * 255).astype(np.uint8)  # 将灰度图转为0~255的uint8
    return local_entropy(gray_uint8, disk(radius))  # 对每个区域计算局部熵

# ----------------------------
# 从局部熵热力图中提取统计特征
# ----------------------------
def extract_local_entropy_stats(local_map):
    return {
        "局部熵均值": np.mean(local_map),
        "局部熵标准差": np.std(local_map),
        "局部熵最大值": np.max(local_map),
        "局部熵90分位数": np.percentile(local_map, 90),
        "局部熵中位数": np.median(local_map),
    }

# ----------------------------
# 计算边缘密度（图像中结构线条所占比例）
# ----------------------------
def compute_edge_density(gray_img):
    gray_uint8 = (gray_img * 255).astype(np.uint8)  # Canny要求uint8输入
    edges = cv2.Canny(gray_uint8, 100, 200)         # Canny边缘检测
    edge_pixels = np.count_nonzero(edges)           # 边缘像素数
    total_pixels = edges.size                       # 图像总像素数
    return edge_pixels / total_pixels               # 返回比例值（0~1）

# ----------------------------
# 主函数：提取图像中所有熵相关特征 + 边缘密度
# ----------------------------
def extract_entropy_features(image_path):
    img = Image.open(image_path).convert("RGB")      # 打开图像并转为RGB
    gray = preprocess_image(img)                     # 灰度化 + 标准化

    H_global = compute_global_entropy(gray)          # 计算全局熵
    local_map = compute_local_entropy_map(gray)      # 局部熵热力图
    edge_density = compute_edge_density(gray)        # 计算边缘密度
    local_stats = extract_local_entropy_stats(local_map)  # 提取局部熵统计特征
    LEDI = local_stats["局部熵均值"] / H_global       # LEDI指标 = 局部熵均值 / 全局熵

    # 汇总所有特征
    features = {
        "全局熵": H_global,
        "LEDI": LEDI,
        "边缘密度": edge_density,
        **local_stats
    }

    return features, gray, local_map

# 使用方法：
# features, gray_img, local_entropy_map = extract_entropy_features("your_image_path.jpg")
# print(features)
# plt.imshow(local_entropy_map, cmap='inferno'); plt.title("局部熵热力图"); plt.axis('off'); plt.show()
# 示例调用（最后几行）

features, gray_img, local_map = extract_entropy_features("1743513358510.png")
print("图像特征如下：")
for k, v in features.items():
    print(f"{k}: {v:.3f}")

plt.imshow(local_map, cmap='inferno')
plt.title("局部熵热力图")
plt.axis('off')
plt.show()

