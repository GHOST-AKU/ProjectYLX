import os
import sys
from typing import List, Optional
import torch
import numpy as np
from PIL import Image
import supervision as sv

sys.path.append("lse/yoloe")
from yoloe.ultralytics import YOLOE


model = YOLOE("lse/yoloe-v8l-seg.pt")
word = ["person"]


def extract_mask_rgb_values(mask_image_path, original_image_path):
    """
    从掩码图像中提取对应区域在原始图像中的RGB值

    Args:
        mask_image_path: 用于获取掩码的图像路径
        original_image_path: 需要提取RGB值的图像路径

    Returns:
        dict: 掩码索引到RGB平均值的映射
    """
    # 加载用于获取掩码的图像
    mask_source_image = Image.open(mask_image_path).convert("RGB")

    # 加载需要提取RGB值的图像
    rgb_source_image = Image.open(original_image_path).convert("RGB")
    rgb_source_np = np.array(rgb_source_image)

    # 加载YOLOE模型并设置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(DEVICE)
    model.set_classes(word, model.get_text_pe(word))

    # 在掩码源图像上进行预测获取掩码
    results = model.predict(mask_source_image, verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])

    # 获取掩码列表
    masks = detections.mask

    if masks is None or len(masks) == 0:
        return {}

    # 提取每个掩码区域的RGB值
    mask_rgb_values = {}

    for i, mask in enumerate(masks):
        # 创建一个布尔掩码数组
        mask_bool = mask.astype(bool)

        # 使用掩码从RGB源图像中提取RGB值
        masked_pixels = rgb_source_np[mask_bool]

        # 计算总和RGB值
        avg_rgb = np.sum(masked_pixels, axis=0)

        # 存储结果
        mask_rgb_values[i + 1] = {
            "R": float(avg_rgb[0]),
            "G": float(avg_rgb[1]),
            "B": float(avg_rgb[2]),
        }

    return mask_rgb_values


def visualize_detections_full(
    image_path: str,
    output: str = "output.png",
) -> Image.Image:
    """
    可视化目标检测结果，包括边界框、掩码和标签

    Args:
        image_path: 输入图像路径
        detections: 目标检测结果

    Returns:
        可视化结果图像
    """
    # 读取原始图像
    image = Image.open(image_path).convert("RGB")
    model.set_classes(word, model.get_text_pe(word))
    results = model.predict(image, verbose=False)

    detections = sv.Detections.from_ultralytics(results[0])
    resolution_wh = image.size
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(
            detections["class_name"], detections.confidence
        )
    ]

    annotated_image = image.copy()
    annotated_image = sv.MaskAnnotator(
        color_lookup=sv.ColorLookup.INDEX, opacity=0.4
    ).annotate(scene=annotated_image, detections=detections)
    annotated_image = sv.BoxAnnotator(
        color_lookup=sv.ColorLookup.INDEX, thickness=thickness
    ).annotate(scene=annotated_image, detections=detections)
    annotated_image = sv.LabelAnnotator(
        color_lookup=sv.ColorLookup.INDEX, text_scale=text_scale, smart_position=True
    ).annotate(scene=annotated_image, detections=detections, labels=labels)

    annotated_image.save(output)

    return annotated_image


def get_detection_results(
    image_path: str,
    model=model,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.5,
) -> dict:
    """
    获取目标检测结果，返回检测到的所有对象及其详细信息

    Args:
        image_path: 输入图像路径
        model: 使用的模型，如果为None则使用默认模型
        confidence_threshold: 置信度阈值，低于此值的检测结果将被过滤
        iou_threshold: IOU阈值，用于NMS过程
        return_image: 是否返回带有标注的图像
        device: 运行设备，为None时自动选择

    Returns:
        包含检测结果的字典，格式为:
        {
            'detections': sv.Detections对象,
            'objects': [
                {
                    'class_id': 类别ID,
                    'class_name': 类别名称,
                    'confidence': 置信度,
                    'box': [x1, y1, x2, y2], # 边界框坐标
                    'mask': 二值掩码数组，  # 实例分割掩码
                    'area': 掩码区域面积,
                    'center': [cx, cy]  # 边界框中心坐标
                },
                ...
            ],
            'summary': {
                'total_objects': 检测到的对象总数,
                'class_counts': {类别名称: 数量, ...},
                'image_size': [宽, 高]
            },
            'annotated_image': 带标注的PIL图像 (仅当return_image=True时)
        }
    """
    import os
    import torch
    import numpy as np
    from PIL import Image
    import supervision as sv

    # 检查图像文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    # 读取图像
    try:
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        image_size = image.size  # 宽, 高
    except Exception as e:
        raise ValueError(f"无法读取或处理图像: {e}")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(DEVICE)
    model.set_classes(word, model.get_text_pe(word))

    # 运行推理
    try:
        results = model.predict(
            image, verbose=False, conf=confidence_threshold, iou=iou_threshold
        )

        detections = sv.Detections.from_ultralytics(results[0])
    except Exception as e:
        raise RuntimeError(f"执行目标检测时出错: {e}")

    return detections


def get_detection_centers(
    detections: sv.Detections,
    image_path: Optional[str] = None,
    output_path: Optional[str] = None,
    visualize: bool = False,
) -> dict:
    """
    从目标检测结果中提取每个目标的中心点

    Args:
        detections: 检测结果对象
        image_path: 原始图像路径，用于可视化
        output_path: 可视化结果保存路径
        visualize: 是否生成可视化结果

    Returns:
        dict: 包含中心点信息的字典，格式为:
        {
            'centers': [
                {
                    'center': [x, y],  # 中心点坐标
                    'class_id': 类别ID,
                    'class_name': 类别名称,  # 如果可用
                    'confidence': 置信度,    # 如果可用
                    'box': [x1, y1, x2, y2]  # 边界框坐标
                },
                ...
            ],
            'image': 带标注的PIL图像 (仅当visualize=True且提供image_path时)
        }
    """
    # 导入必要的库
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import cv2

    # 检查detections是否为空
    if not hasattr(detections, "xyxy") or len(detections.xyxy) == 0:
        return {"centers": [], "image": None}

    # 准备返回结果
    centers_info = []

    # 计算每个检测框的中心点
    for i, box in enumerate(detections.xyxy):
        # 提取边界框坐标
        x1, y1, x2, y2 = box

        # 计算中心点
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 创建中心点信息字典
        center_info = {
            "center": [float(center_x), float(center_y)],
            "box": [float(x1), float(y1), float(x2), float(y2)],
        }

        # 如果有类别ID信息
        if hasattr(detections, "class_id") and i < len(detections.class_id):
            center_info["class_id"] = int(detections.class_id[i])

        # 如果有类别名称信息
        if hasattr(detections, "class_name") and i < len(detections.class_name):
            center_info["class_name"] = detections.class_name[i]

        # 如果有置信度信息
        if hasattr(detections, "confidence") and i < len(detections.confidence):
            center_info["confidence"] = float(detections.confidence[i])

        centers_info.append(center_info)

    result = {"centers": centers_info}

    # 如果需要可视化并提供图像路径
    if visualize and image_path:
        try:
            # 读取原始图像
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)

            # 尝试加载字体
            try:
                font = ImageFont.truetype("arial.ttf", 15)  # Windows默认字体
            except:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", 15)  # Linux常见字体
                except:
                    font = ImageFont.load_default()

            # 绘制每个中心点
            for center_info in centers_info:
                # 获取中心点坐标
                cx, cy = center_info["center"]

                # 绘制中心点（红色十字标记）
                cross_size = 10
                draw.line(
                    (cx - cross_size, cy, cx + cross_size, cy),
                    fill=(255, 0, 0),
                    width=2,
                )
                draw.line(
                    (cx, cy - cross_size, cx, cy + cross_size),
                    fill=(255, 0, 0),
                    width=2,
                )

                # 绘制点的编号或类别
                text = f"ID:{centers_info.index(center_info)}"
                if "class_name" in center_info:
                    text = f"{center_info['class_name']}"
                if "confidence" in center_info:
                    text += f" {center_info['confidence']:.2f}"

                # 添加中心点坐标
                text += f"\n({int(cx)},{int(cy)})"

                # 绘制文本
                draw.text((cx + 5, cy + 5), text, fill=(255, 0, 0), font=font)

            # 如果指定了输出路径，保存图像
            if output_path:
                image.save(output_path)
                print(f"已保存标注中心点的图像到: {output_path}")

            # 将图像添加到结果中
            result["image"] = image

        except Exception as e:
            print(f"可视化中心点时出错: {e}")

    return result


if __name__ == "__main__":
    # 图像路径
    depth_image_path = "output_colored.png"  # 彩色深度图路径
    segmentation_image_path = r"深度图\7\1743513385553.png"  # 分割模型使用的原始图像
    output = "seg-output.jpg"
    # visualize_detections_full(segmentation_image_path,output)
    image_path = "20250329221858.png"
    detections = get_detection_results(image_path, model)

    centers_result = get_detection_centers(
        detections,
        image_path=image_path,
        output_path="detection_centers.png",
        visualize=True,
    )

    # 打印中心点信息
    print(f"检测到 {len(centers_result['centers'])} 个目标中心点:")
    for i, center_info in enumerate(centers_result["centers"]):
        center_str = f"点 {i}: 坐标 ({center_info['center'][0]:.1f}, {center_info['center'][1]:.1f})"
        if "class_name" in center_info:
            center_str += f", 类别: {center_info['class_name']}"
        if "confidence" in center_info:
            center_str += f", 置信度: {center_info['confidence']:.2f}"
        print(center_str)

    # 显示结果图像
    if centers_result["image"]:
        centers_result["image"].show()
