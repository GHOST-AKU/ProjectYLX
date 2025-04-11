
import os
import pandas as pd
from entropy_analysis import extract_entropy_features
from tqdm import tqdm
import matplotlib.pyplot as plt

# 根目录：请根据你的实际路径调整
base_path = r"C:\Users\GHOST_AKU\Desktop\input"
output_csv = "熵特征提取结果.csv"
output_heatmap_dir = "output_heatmaps"

# 创建输出文件夹
os.makedirs(output_heatmap_dir, exist_ok=True)

# 输出结果列表
results = []

# 要处理的两个图像类型文件夹
modes = ["original", "deepth"]

for mode in modes:
    root_path = os.path.join(base_path, mode)
    for label_folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, label_folder)
        if not os.path.isdir(folder_path):
            continue

        for filename in tqdm(os.listdir(folder_path), desc=f"🔍 {mode} | 分类 {label_folder}"):
            if not filename.lower().endswith((".jpg", ".png")):
                continue

            image_path = os.path.join(folder_path, filename)
            try:
                features, gray_img, local_map = extract_entropy_features(image_path)

                # 加入基本信息
                features["图像路径"] = image_path
                features["类别编号"] = label_folder
                features["图像来源"] = mode
                results.append(features)

                # 保存局部熵热力图为图像文件
                heatmap_filename = f"{mode}_{label_folder}_{os.path.splitext(filename)[0]}_heatmap.jpg"
                heatmap_path = os.path.join(output_heatmap_dir, heatmap_filename)

                plt.imshow(local_map, cmap='inferno')
                plt.axis('off')
                plt.title("局部熵热力图")
                plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
                plt.close()

            except Exception as e:
                print(f"❌ 错误处理文件 {image_path}: {e}")

# 保存为 CSV 文件
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"✅ 所有图像处理完成！特征保存在：{output_csv}")
print(f"🖼️ 局部熵热力图已保存到文件夹：{output_heatmap_dir}")
