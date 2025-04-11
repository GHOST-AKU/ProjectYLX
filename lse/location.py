import cv2
import numpy as np
import os

from depth_interface import loadmodel, depth_predict
from objectdetect import get_detection_results, get_detection_centers

model = loadmodel()

depth_predict(model, 'lse/assert/20250329221905.png', is_save=True)

results = get_detection_results('lse/assert/20250329221905.png')
centers = get_detection_centers(results)

FX, FY = 2900.0, 2900.0  # iPhone 8 估算的焦距（像素单位）
CX, CY = 2016, 1512  # iPhone 8 图像中心点 (4032x3024)

# ============  加载深度图  ============
DEPTH_PATH = r"lse\assert\output_depth.png"
rgb_path = r"lse\assert\20250329221905.png"

def load_depth_map(path):
    """加载深度图，将其转换为深度值(米)"""
    depth_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # 读取灰度深度图
    if depth_img is None:
        raise FileNotFoundError(f"无法加载深度图: {path}")

    # 根据深度图类型进行处理
    if depth_img.dtype == np.uint8:
        # 8位灰度图
        depth_img = depth_img.astype(np.float32) / 255.0  # 归一化到 0-1
        depth_img *= 10.0  # 假设深度范围 0-10 米
    elif depth_img.dtype == np.uint16:
        # 16位深度图
        depth_img = depth_img.astype(np.float32) / 65535.0  # 归一化到 0-1
        depth_img *= 10.0  # 假设深度范围 0-10 米
    
    return depth_img


def load_color_image(depth_path:str,rgb_image:str):
    """尝试加载深度图对应的彩色图像"""
    # 尝试查找与深度图相关联的彩色图像
    if os.path.exists(rgb_image):
        color_img = cv2.imread(rgb_image)
    return color_img


# ============  像素坐标转换为3D坐标  ============
def convert_to_3d(x, y, depth_map):
    """将图像上的像素坐标转换为3D世界坐标"""
    # 确保x和y在深度图范围内
    h, w = depth_map.shape[:2]
    x, y = max(0, min(x, w-1)), max(0, min(y, h-1))
    
    # 获取深度值
    z = depth_map[y, x]  # 读取深度值（单位：米）
    
    # 确保z是标量
    if isinstance(z, np.ndarray):
        z = float(z.item()) if z.size == 1 else float(z.mean())
    
    # 转换到相机坐标系
    x_3d = (x - CX) * z / FX
    y_3d = (y - CY) * z / FY
    
    return np.array([x_3d, y_3d, z])


# ============  计算两点间3D距离  ============
def calculate_distance(p1, p2):
    """计算两个3D点之间的欧氏距离"""
    return np.linalg.norm(p1 - p2)


# ============  计算物体中心点距离并可视化  ============
def visualize_center_distances(centers_data, depth_path, output_path="center_distances.jpg"):
    """
    计算物体中心点之间的距离，在图像上可视化并保存结果
    
    参数:
        centers_data: 包含物体中心点信息的字典
        depth_path: 深度图路径
        output_path: 输出图像保存路径
    """
    # 加载深度图
    depth_map = load_depth_map(depth_path)
    
    # 加载彩色图像作为背景
    color_img = load_color_image(depth_path,rgb_path)
    if color_img is None:
        # 如果无法加载彩色图像，则创建一个空白的彩色图像
        h, w = depth_map.shape[:2]
        color_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 调整相机参数以适应图像尺寸
    global CX, CY
    h, w = depth_map.shape[:2]
    CX = w // 2
    CY = h // 2
    
    # 存储物体中心点的3D坐标
    centers_3d = []
    
    # 为不同类别分配不同颜色
    class_colors = {
        'person': (0, 0, 255),     # 红
        'bicycle': (0, 255, 0),    # 绿
        'car': (255, 0, 0),        # 蓝
        'motorcycle': (255, 255, 0),  # 青
        'default': (255, 255, 255)  # 白色 (默认)
    }
    
    # 处理每个中心点
    for i, center_info in enumerate(centers_data['centers']):
        x, y = center_info['center']
        x, y = int(round(x)), int(round(y))
        
        # 确保坐标在图像范围内
        x = max(0, min(x, w-1))
        y = max(0, min(y, h-1))
        
        # 获取该点的深度值并转换为3D坐标
        p_3d = convert_to_3d(x, y, depth_map)
        
        # 获取该点类别
        class_name = center_info.get('class_name', 'unknown')
        confidence = center_info.get('confidence', 1.0)
        
        # 保存中心点信息
        centers_3d.append({
            'index': i,
            'pixel': (x, y),
            '3d': p_3d,
            'class_name': class_name,
            'confidence': confidence
        })
        
        # 获取物体类别对应的颜色
        color = class_colors.get(class_name, class_colors['default'])
        
        # 在图像上绘制中心点
        cv2.circle(color_img, (x, y), 5, color, -1)
        
        # 添加标签
        label = f"{i}: {class_name}"
        if confidence < 1.0:
            label += f" {confidence:.2f}"
        
        # 使用黑色描边的白色文字以增强可读性
        cv2.putText(color_img, label, (x+10, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(color_img, label, (x+10, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 计算并绘制中心点之间的距离
    distances = []
    for i, center1 in enumerate(centers_3d):
        for j, center2 in enumerate(centers_3d):
            if i >= j:  # 避免重复计算
                continue
                
            # 计算3D距离
            distance = calculate_distance(center1['3d'], center2['3d'])
            
            # 保存距离信息
            distances.append({
                'from': i,
                'to': j,
                'distance': distance,
                'from_class': center1['class_name'],
                'to_class': center2['class_name']
            })
            
            # 在图像上绘制连接线
            cv2.line(color_img, center1['pixel'], center2['pixel'], (0, 255, 255), 1)
            
            # 计算连接线中点
            mid_x = (center1['pixel'][0] + center2['pixel'][0]) // 2
            mid_y = (center1['pixel'][1] + center2['pixel'][1]) // 2
            
            # 在连接线中点显示距离
            distance_text = f"{distance:.2f}m"
            
            # 使用黑色描边的白色文字以增强可读性
            cv2.putText(color_img, distance_text, (mid_x, mid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(color_img, distance_text, (mid_x, mid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 打印中心点信息
    print(f"检测到 {len(centers_3d)} 个目标中心点:")
    for center in centers_3d:
        center_str = f"点 {center['index']}: 坐标 ({center['pixel'][0]:.1f}, {center['pixel'][1]:.1f})"
        if 'class_name' in center:
            center_str += f", 类别: {center['class_name']}"
        if 'confidence' in center:
            center_str += f", 置信度: {center['confidence']:.2f}"
        print(center_str)
    
    # 打印距离信息
    print("\n物体之间的距离:")
    for dist in distances:
        print(f"物体 {dist['from']} 和物体 {dist['to']} 之间的距离: {dist['distance']:.2f}米")
    
    # 保存结果图像
    cv2.imwrite(output_path, color_img)
    print(f"结果图像已保存到: {output_path}")
    
    return color_img, centers_3d, distances


# ============  保存结果功能  ============
def save_result(image, filename="distance_measurement_result.jpg"):
    """保存测量结果图像"""
    cv2.imwrite(filename, image)
    print(f"结果已保存到: {filename}")


# ============  运行  ============
if __name__ == "__main__":
    try:
        # 直接计算物体中心点距离并可视化
        visualize_center_distances(centers, DEPTH_PATH, "object_distances.jpg")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()