"""
CLIP模型基础演示
================

本文件演示了OpenAI CLIP模型的基础功能，包括：
1. 零样本图像分类 - 在没有训练数据的情况下对图像进行分类
2. 图文检索 - 根据文本描述查找最相关的图像

CLIP（Contrastive Language-Image Pre-training）是一个多模态模型，
能够理解图像和文本之间的关系，广泛应用于图像检索、分类等任务。
"""

import torch
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

# 从配置文件中导入路径配置
from config import DATA_IMAGES, DATA_RESULTS


# ====================
# 1. 模型初始化和加载
# ====================

print("正在加载 CLIP 模型...")
# 加载CLIP模型的处理器，负责预处理图像和文本数据
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 加载CLIP模型的预训练权重
# AutoModelForZeroShotImageClassification用于零样本图像分类任务
model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-base-patch32")

# 设置模型为评估模式，禁用dropout等训练时特有的层
model.eval()

# 自动选择计算设备：如果有GPU可用则使用CUDA，否则使用CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 将模型移动到指定的计算设备上
model.to(device)
print(f"模型已加载到: {device}")

# ====================
# 2. 零样本分类功能
# ====================

def zero_shot_classification(image_path, labels):
    """
    零样本图像分类功能
    
    零样本分类是指在没有任何针对特定类别的训练数据的情况下，
    直接使用预训练的CLIP模型对图像进行分类。
    
    Args:
        image_path (str): 图像文件路径
        labels (list): 候选分类标签列表
    
    Returns:
        dict: 包含分类结果的字典，包括最高概率标签、概率值等
    
    工作原理：
    1. 将图像和所有候选标签转换为CLIP的输入格式
    2. 计算图像与每个标签的相似度分数
    3. 使用softmax将相似度转换为概率分布
    4. 返回概率最高的标签作为预测结果
    """
    
    # 加载图像并转换为RGB格式（确保兼容性）
    image = Image.open(image_path).convert("RGB")

    # 使用processor预处理图像和文本
    # padding=True确保批次中所有样本的长度一致
    # return_tensors="pt"返回PyTorch张量格式
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    
    # 将输入数据移动到模型所在的设备（GPU/CPU）
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 禁用梯度计算，提高推理速度并节省内存
    with torch.no_grad():
        # 前向传播计算
        outputs = model(**inputs)
        # 获取图像-文本相似度分数
        # logits_per_image shape: [1, num_labels]
        logits = outputs.logits_per_image
        # 使用softmax转换为概率分布
        probs = logits.softmax(dim=1).cpu().numpy().flatten()

    # 打印详细的分类结果
    print(f"\n图像: {os.path.basename(image_path)}")
    for label, prob in zip(labels, probs):
        print(f"  {label}: {prob:.3f}")

    # 返回结构化的结果，便于后续分析和保存
    return {
        "image": image_path,                           # 图像路径
        "top_label": labels[probs.argmax()],          # 概率最高的标签
        "top_prob": float(probs.max()),               # 最高概率值
        "all_probs": {label: float(prob) for label, prob in zip(labels, probs)}  # 所有标签的概率
    }


# ====================
# 3. 图文检索功能
# ====================

def text_image_retrieval(text, image_folder, top_k=3):
    """
    基于文本的图像检索功能
    
    根据给定的文本描述，从图像库中检索出最相关的图像。
    这是CLIP模型的核心应用之一。
    
    Args:
        text (str): 查询文本描述
        image_folder (str): 图像文件夹路径
        top_k (int): 返回最相似图像的数量
    
    Returns:
        dict: 包含查询文本和检索结果的字典
    
    工作原理：
    1. 加载文件夹中的所有图像
    2. 使用CLIP分别提取文本和图像的特征向量
    3. 计算文本特征与每个图像特征的余弦相似度
    4. 返回相似度最高的前k张图像
    """
    
    # 获取文件夹中所有图像文件
    images = []
    image_paths = []

    # 遍历文件夹，收集支持的图像格式
    for fname in os.listdir(image_folder):
        # 支持常见的图像格式：PNG、JPG、JPEG
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(image_folder, fname)
            image_paths.append(path)
            # 加载图像并转换为RGB格式
            images.append(Image.open(path).convert("RGB"))

    # 检查是否找到图像文件
    if not images:
        print("没有找到图像！")
        return

    # 使用processor分别处理文本和图像
    # 文本输入：转换为文本特征
    text_inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    # 图像输入：批量处理所有图像
    image_inputs = processor(images=images, return_tensors="pt").to(device)

    # 特征提取和相似度计算
    with torch.no_grad():
        # 提取文本的特征向量
        text_features = model.get_text_features(**text_inputs)
        # 提取所有图像的特征向量
        image_features = model.get_image_features(**image_inputs)

        # 特征向量归一化：确保余弦相似度计算的准确性
        # L2归一化将向量缩放为单位长度
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 计算余弦相似度矩阵
        # matmul操作计算文本特征与所有图像特征的点积
        # squeeze(dim=0)移除batch维度，避免后续计算错误
        similarities = (text_features @ image_features.T).squeeze(dim=0)

    # 找出最相似的图像
    # 限制返回数量不超过实际图像数量
    top_k = min(top_k, len(images))
    
    # 使用torch.topk找出相似度最高的k个图像的索引
    top_indices = torch.topk(similarities, top_k).indices

    # 维度安全检查：确保索引是一维张量
    # 避免在只有一张图像时topk返回标量导致后续计算错误
    if top_indices.ndim == 0:
        top_indices = top_indices.unsqueeze(0)

    # 打印检索结果
    print(f"\n查询文本: '{text}'")
    print(f"检索结果（从 {len(images)} 张图像中）:")

    results = []
    # 处理每个检索结果
    for i, idx in enumerate(top_indices):
        # 兼容处理：确保索引是Python整数类型
        idx = int(idx) if isinstance(idx, torch.Tensor) else idx
        
        # 获取对应的相似度分数
        similarity = float(similarities[idx])
        # 获取对应的图像路径
        path = image_paths[idx]
        
        # 打印结果信息
        print(f"  {i + 1}. {os.path.basename(path)} (相似度: {similarity:.3f})")
        
        # 构建结果记录
        results.append({
            "image": path,              # 图像路径
            "similarity": similarity,   # 相似度分数
            "rank": i + 1              # 排名（1为最相似）
        })

    return {"query": text, "results": results}


# ====================
# 4. 结果可视化功能
# ====================

def create_visualization_comparison(image_path, labels, probs, output_path, task_type="classification"):
    """
    创建原图与输出结果的对比可视化图像
    
    Args:
        image_path: 原始图像路径
        labels: 分类标签列表
        probs: 对应概率列表
        output_path: 输出图像保存路径
        task_type: 任务类型（classification/retrieval）
    """
    original_image = Image.open(image_path).convert("RGB")
    
    width, height = original_image.size
    result_width = width * 2
    result_height = height
    
    result_image = Image.new('RGB', (result_width, result_height), (255, 255, 255))
    result_image.paste(original_image, (0, 0))
    
    draw = ImageDraw.Draw(result_image)
    
    try:
        font_size = int(min(width, height) * 0.035)
        title_font = ImageFont.truetype("arial.ttf", font_size)
        label_font = ImageFont.truetype("arial.ttf", int(font_size * 0.8))
        prob_font = ImageFont.truetype("arial.ttf", int(font_size * 0.7))
    except:
        try:
            font_size = int(min(width, height) * 0.035)
            title_font = ImageFont.truetype("simhei.ttf", font_size)
            label_font = ImageFont.truetype("simhei.ttf", int(font_size * 0.8))
            prob_font = ImageFont.truetype("simhei.ttf", int(font_size * 0.7))
        except:
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()
            prob_font = ImageFont.load_default()
    
    info_x = width + 20
    info_y = 20
    
    draw.text((info_x, info_y), "CLIP Zero-Shot Classification", fill=(0, 0, 0), font=title_font)
    info_y += font_size + 30
    
    sorted_indices = np.argsort(probs)[::-1]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_probs = [probs[i] for i in sorted_indices]
    
    bar_start_x = info_x
    bar_start_y = info_y + font_size
    bar_height = int(height * 0.7 / len(labels))
    max_bar_width = int(width * 0.35)
    
    max_prob = max(sorted_probs) if max(sorted_probs) > 0 else 1.0
    
    for i, (label, prob) in enumerate(zip(sorted_labels, sorted_probs)):
        label_y = bar_start_y + i * bar_height
        
        bar_width = int((prob / max_prob) * max_bar_width) if max_prob > 0 else 0
        
        bar_color = (int(50 + 200 * (1 - i / len(labels))), 
                     int(100 + 155 * (i / len(labels))), 
                     255)
        draw.rectangle([bar_start_x, label_y, bar_start_x + bar_width, label_y + bar_height - 2], 
                      fill=bar_color)
        
        draw.text((bar_start_x + bar_width + 5, label_y), 
                 f"{prob:.3f}", fill=(0, 0, 0), font=prob_font)
    
    info_y = bar_start_y + len(labels) * bar_height + 20
    
    result_image.save(output_path, quality=95)
    print(f"可视化对比图已保存至: {output_path}")


def save_classification_result(image_path, labels, all_probs, output_dir):
    """保存分类结果到JSON文件"""
    result = {
        "image": str(image_path),
        "top_label": labels[all_probs.argmax()],
        "top_prob": float(all_probs.max()),
        "all_predictions": {label: float(prob) for label, prob in zip(labels, all_probs)}
    }
    
    output_path = os.path.join(output_dir, "classification_result.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"分类结果已保存至: {output_path}")
    return output_path


def create_retrieval_comparison(query_text, results, image_folder, output_path):
    """创建图文检索结果对比可视化"""
    if not results:
        return
    
    sample_image = Image.open(results[0]["image"]).convert("RGB")
    img_width, img_height = sample_image.size
    
    result_count = len(results)
    total_width = img_width * (result_count + 1)
    total_height = img_height
    
    result_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    
    draw = ImageDraw.Draw(result_image)
    
    try:
        font_size = int(min(img_width, img_height) * 0.04)
        title_font = ImageFont.truetype("arial.ttf", font_size)
        label_font = ImageFont.truetype("arial.ttf", int(font_size * 0.7))
    except:
        try:
            font_size = int(min(img_width, img_height) * 0.04)
            title_font = ImageFont.truetype("simhei.ttf", font_size)
            label_font = ImageFont.truetype("simhei.ttf", int(font_size * 0.7))
        except:
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()
    
    draw.text((20, 20), f"Query: '{query_text}'", fill=(0, 0, 150), font=title_font)
    
    for i, result in enumerate(results):
        img = Image.open(result["image"]).convert("RGB")
        x_offset = img_width * (i + 1)
        result_image.paste(img, (x_offset, 0))
        
        label_y = 20
        label_text = f"Rank {result['rank']}"
        draw.text((x_offset + 20, label_y), label_text, fill=(0, 0, 0), font=title_font)
        
        score_y = label_y + font_size + 10
        score_text = f"Sim: {result['similarity']:.3f}"
        draw.text((x_offset + 20, score_y), score_text, fill=(0, 100, 0), font=label_font)
    
    result_image.save(output_path, quality=95)
    print(f"检索结果对比图已保存至: {output_path}")


def save_retrieval_result(query_text, results, output_dir):
    """保存检索结果到JSON文件"""
    result = {
        "query": query_text,
        "results": results
    }
    
    output_path = os.path.join(output_dir, "retrieval_result.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"检索结果已保存至: {output_path}")
    return output_path


# ====================
# 5. 主程序入口
# ====================

if __name__ == "__main__":
    # 获取图像数据目录路径
    data_images = DATA_IMAGES
    
    # 创建CLIP演示结果保存目录
    clip_output_dir = DATA_RESULTS / "clip_demo"
    os.makedirs(clip_output_dir, exist_ok=True)
    
    print("=== CLIP 基础复现演示 ===")

    # 演示1: 零样本图像分类
    print("\n1. 零样本图像分类:")

    # 构建测试图像的完整路径
    test_image = data_images / "0.jpg"
    
    # 检查测试图像是否存在
    if os.path.exists(test_image):
        # 定义候选分类标签
        labels = ["a cat", "a dog", "a car", "a tree", "a person", "a phone"]
        
        # 执行零样本分类
        result = zero_shot_classification(test_image, labels)
        
        # 保存结果到data/results
        save_classification_result(test_image, labels, 
                                  np.array([result["all_probs"][label] for label in labels]),
                                  clip_output_dir)
        
        # 创建可视化对比图
        probs = [result["all_probs"][label] for label in labels]
        create_visualization_comparison(
            test_image, labels, probs,
            os.path.join(clip_output_dir, "classification_comparison.png"),
            task_type="classification"
        )
    else:
        # 如果图像不存在，提供使用指导
        print(f"测试图像不存在: {test_image}")
        print("请先放置一张测试图像在 data/images/ 文件夹中")

    # 演示2: 图文检索
    print("\n2. 图文检索:")
    
    # 检查图像文件夹是否有效
    if os.path.exists(data_images) and len(os.listdir(data_images)) > 0:
        # 执行文本检索图像任务
        retrieval_result = text_image_retrieval("a person", data_images, top_k=3)
        
        if retrieval_result:
            # 保存检索结果
            save_retrieval_result(retrieval_result["query"], retrieval_result["results"], clip_output_dir)
            
            # 创建检索结果可视化
            create_retrieval_comparison(
                retrieval_result["query"],
                retrieval_result["results"],
                data_images,
                os.path.join(clip_output_dir, "retrieval_comparison.png")
            )
    else:
        print("图像文件夹为空，请先添加一些图像")

    print("\n=== 演示完成 ===")
    print(f"所有结果已保存至: {clip_output_dir}")