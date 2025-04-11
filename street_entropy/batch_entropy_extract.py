import os
import pandas as pd
from entropy_analysis import extract_entropy_features  # 你之前的熵分析函数文件
from tqdm import tqdm

# 根目录：改成你自己的绝对路径
base_path = r"C:\Users\GHOST_AKU\Desktop\input"

# 输出结果列表
results = []

# 扫描 original 文件夹
original_root = os.path.join(base_path, "original")

# 遍历每个子文件夹（比如 1、2、3...）
for label_folder in os.listdir(original_root):
    folder_path = os.path.join(original_root, label_folder)
    if not os.path.isdir(folder_path):
        continue

    for filename in tqdm(os.listdir(folder_path), desc=f"📷 Processing folder {label_folder}"):
        if not filename.lower().endswith((".jpg", ".png")):
            continue

        image_path = os.path.join(folder_path, filename)
        try:
            features, _, _ = extract_entropy_features(image_path)
            features["图像路径"] = image_path
            features["类别编号"] = label_folder
            results.append(features)
        except Exception as e:
            print(f"❌ 错误处理文件 {image_path}: {e}")

# 转为 DataFrame 并保存
df = pd.DataFrame(results)
df.to_csv("熵特征提取结果.csv", index=False, encoding="utf-8-sig")

print("✅ 所有图像处理完成！特征保存在：熵特征提取结果.csv")
