import os
from PIL import Image
import torch
from loguru import logger
from tqdm import tqdm
from ZoeDepth.zoedepth.utils.misc import colorize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.debug(f"Using device: {DEVICE}")


def loadmodel() -> torch.nn.Module:
    repo = "isl-org/ZoeDepth"
    torch.hub.set_dir("ZoeDepth")
    model_zoe_n = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
    zoe = model_zoe_n.to(DEVICE)
    return zoe


def depth_predict(
    zoe: torch.nn.Module, image: str, is_save: bool = False
) -> torch.Tensor:
    image_PIL = Image.open(image).convert("RGB")  # load
    depth_tensor = zoe.infer_pil(image_PIL, output_type="tensor")  # as torch tensor
    if is_save:
        colored = colorize(depth_tensor)
        fpath_colored = "output_colored.png"
        Image.fromarray(colored).save(fpath_colored)

    return depth_tensor


def process_directory(
    zoe: torch.nn.Module,
    input_dir: str,
    output_dir: str,
    is_save: bool = True,
    extensions: tuple = (".png", ".jpg", ".jpeg", ".bmp", ".webp"),
) -> None:
    """
    递归处理目录下的所有图像文件，生成深度图并保持原始文件夹结构

    Args:
        zoe: 加载好的ZoeDepth模型
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        is_save: 是否保存深度图，默认为True
        extensions: 要处理的图像文件扩展名元组
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取当前目录下的所有文件和文件夹
    items = os.listdir(input_dir)

    # 用于统计处理结果
    stats = {"processed": 0, "skipped": 0, "errors": 0, "dirs": 0}

    # 创建进度条
    progress = tqdm(total=len(items), desc=f"处理 {os.path.basename(input_dir)}")

    # 处理每个项目
    for item in items:
        input_path = os.path.join(input_dir, item)
        output_path = os.path.join(output_dir, item)

        # 如果是目录，递归处理
        if os.path.isdir(input_path):
            stats["dirs"] += 1
            # 创建对应的输出目录
            os.makedirs(output_path, exist_ok=True)
            # 递归处理子目录
            sub_stats = process_directory(
                zoe, input_path, output_path, is_save, extensions
            )
            # 合并统计数据
            for key in stats:
                stats[key] += sub_stats.get(key, 0)

        # 如果是图像文件，处理它
        elif os.path.isfile(input_path) and input_path.lower().endswith(extensions):
            try:
                # 处理图像
                depth_tensor = depth_predict(zoe, input_path, False)  # 不在这里保存

                if is_save:
                    # 生成深度图
                    colored = colorize(depth_tensor)

                    # 构建输出文件路径，保持原始文件名但添加_depth后缀
                    filename, extension = os.path.splitext(item)
                    output_file = f"{filename}_depth{extension}"
                    output_file_path = os.path.join(output_dir, output_file)

                    # 保存彩色深度图
                    Image.fromarray(colored).save(output_file_path)

                stats["processed"] += 1

            except Exception as e:
                logger.error(f"处理 {input_path} 时出错: {str(e)}")
                stats["errors"] += 1
        else:
            # 跳过非图像文件
            stats["skipped"] += 1

        # 更新进度条
        progress.update(1)

    # 关闭进度条
    progress.close()

    # 在完成处理当前目录后，打印统计信息
    if input_dir == output_dir.replace("_depth", ""):  # 只在顶层目录打印
        logger.info(f"处理完成！统计信息:")
        logger.info(f"  - 已处理图像: {stats['processed']} 个")
        logger.info(f"  - 已跳过文件: {stats['skipped']} 个")
        logger.info(f"  - 处理出错: {stats['errors']} 个")
        logger.info(f"  - 递归处理目录: {stats['dirs']} 个")

    return stats


model = loadmodel()

process_directory(model, "深度图", "output")
