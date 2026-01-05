"""
================================================================================
CLIP模型基础演示脚本
================================================================================

LIP模型调用的基本方法。

主要功能模块：
1. 零样本图像分类 - 使用pipeline一行代码完成图像分类
2. 图文检索 - 使用CLIP特征提取进行相似度计算和图像检索
3. 结果可视化 - 将分类和检索结果保存为图像文件

================================================================================
"""

# ==============================================================================
# 第一部分：环境配置和库导入
# ==============================================================================

# 导入标准库
import os

# 设置OpenMP库兼容性选项
# 解决某些系统上可能出现的多重库冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 导入配置模块
from cxc.config import DATA_IMAGES, DATA_RESULTS

# 导入公共功能模块中的结果保存和可视化函数
from vqa_common import (
    save_classification_result,    # 保存分类结果到JSON
    save_retrieval_result,         # 保存检索结果到JSON
    create_visualization_comparison,  # 创建分类可视化对比图
    create_retrieval_comparison    # 创建检索可视化对比图
)
from transformers import pipeline, AutoProcessor, AutoModelForZeroShotImageClassification
import torch
from PIL import Image
import numpy as np

# ==============================================================================
# 第二部分：模型初始化
# ==============================================================================

# 创建零样本图像分类pipeline：内部自动加载了对应的processor和model
classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
print("CLIP分类器加载完成")

# 初始化CLIP模型的处理器（Processor）
# 负责将图像和文本转换为模型所需的输入格式
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 加载CLIP预训练模型
# AutoModelForZeroShotImageClassification是专门用于零样本分类的模型类
clip_model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-base-patch32")

# 设置模型为评估模式
# 评估模式会关闭Dropout等训练专用层，确保推理结果稳定
clip_model.eval()

# 自动检测并选择计算设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 将模型移动到指定设备
clip_model.to(device)
print(f"CLIP模型已加载到: {device}")


# ==============================================================================
# 第三部分：零样本图像分类函数：使用pipline
# ==============================================================================

def zero_shot_classification(image_path, labels):
    """
    使用CLIP pipeline进行零样本图像分类
    
    该函数接收一张图像和一组候选标签，返回每个标签的预测概率。
    CLIP模型通过计算图像与各标签文本的相似度来进行分类。
    
    工作原理：
    1. 将图像编码为特征向量
    2. 将每个标签文本编码为特征向量
    3. 计算图像特征与各文本特征的余弦相似度
    4. 通过softmax将相似度转换为概率分布
    
    Args:
        image_path (str): 图像文件的路径
        labels (list): 候选标签列表，如 ["猫", "狗", "车"]
        
    Returns:
        dict: 分类结果字典，包含以下字段：
            - image: 输入图像路径
            - top_label: 概率最高的标签
            - top_prob: 最高概率值
            - all_probs: 所有标签的概率字典

    """
    # 确保路径是字符串类型
    if not isinstance(image_path, str):
        image_path = str(image_path)
        
    # 调用pipeline进行分类
    # pipeline自动处理图像预处理和模型推理
    result = classifier(image_path, candidate_labels=labels)
    
    # 提取概率列表
    probs = [r["score"] for r in result]
    
    # 打印详细结果
    print(f"\n图像: {os.path.basename(image_path)}")
    for label, prob in zip(labels, probs):
        print(f"  {label}: {prob:.3f}")
        
    # 构建并返回结果字典
    return {
        "image": image_path,
        "top_label": result[0]["label"],
        "top_prob": result[0]["score"],
        "all_probs": {r["label"]: r["score"] for r in result}
    }


# ==============================================================================
# 第四部分：图文检索函数：使用processor和clip_model
# ==============================================================================

def text_image_retrieval(text, image_folder, top_k=3):
    """
    使用CLIP进行图文检索
    
    该函数根据给定的文本查询，在图像文件夹中检索最相关的图像。
    CLIP模型将文本和图像都编码到同一个向量空间，
    通过计算余弦相似度来评估文本与图像的匹配程度。
    
    工作原理：
    1. 将查询文本编码为特征向量
    2. 读取文件夹中所有图像并编码为特征向量
    3. 计算文本特征与各图像特征的余弦相似度
    4. 返回相似度最高的top_k张图像
    
    Args:
        text (str): 查询文本，如 "a person"
        image_folder (str): 图像文件夹路径
        top_k (int): 返回的检索结果数量，默认3
        
    Returns:
        dict or None: 检索结果字典，包含以下字段：
            - query: 查询文本
            - results: 检索结果列表，每个元素包含：
                - image: 图像路径
                - similarity: 相似度分数
                - rank: 排名
            如果没有找到图像，返回None

    """
    # 扫描文件夹中的图像文件
    images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 检查是否找到图像
    if not images:
        print("没有找到图像！")
        return None

    # 构建完整的图像路径列表
    image_paths = [os.path.join(image_folder, f) for f in images]
    
    # 使用PIL打开所有图像并转换为RGB模式
    pil_images = [Image.open(p).convert("RGB") for p in image_paths]

    # 禁用梯度计算
    with torch.no_grad():
        # 处理文本输入
        # return_tensors="pt" 返回PyTorch张量，padding=True 将不同长度的文本填充到相同长度
        text_inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
        
        # 获取文本特征向量
        text_features = clip_model.get_text_features(**text_inputs)
        
        # L2归一化，使特征向量的模为1
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 处理图像输入
        image_inputs = processor(images=pil_images, return_tensors="pt").to(device)
        
        # 获取图像特征向量
        image_features = clip_model.get_image_features(**image_inputs)
        
        # L2归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 计算文本与所有图像的相似度
        # 通过矩阵乘法一次计算所有相似度，归一化后，余弦相似度等于向量点积
        similarities = (text_features @ image_features.T).squeeze(0)

    # 限制返回数量不超过图像总数
    top_k = min(top_k, len(images))
    
    # 获取相似度最高的top_k个图像的索引
    top_indices = torch.topk(similarities, top_k).indices.tolist()

    # 构建检索结果
    results = []
    print(f"\n查询文本: '{text}'")
    print(f"检索结果（从 {len(images)} 张图像中）:")
    
    for i, idx in enumerate(top_indices):
        similarity = float(similarities[idx])
        path = image_paths[idx]
        results.append({"image": path, "similarity": similarity, "rank": i + 1})
        print(f"  {i + 1}. {os.path.basename(path)} (相似度: {similarity:.3f})")

    return {"query": text, "results": results}


# ==============================================================================
# 第五部分：主程序入口
# ==============================================================================

if __name__ == "__main__":
    """
    主程序执行流程：
    1. 创建输出目录
    2. 执行零样本图像分类演示
    3. 执行图文检索演示
    4. 保存所有结果
    """
    
    # 创建CLIP演示结果的输出目录
    clip_output_dir = DATA_RESULTS / "clip_demo"
    os.makedirs(clip_output_dir, exist_ok=True)
    
    print("=" * 60)
    print("CLIP 基础复现演示")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 演示1：零样本图像分类
    # -------------------------------------------------------------------------
    print("\n1. 零样本图像分类:")
    
    # 选择测试图像
    test_image = DATA_IMAGES / "0.jpg"
    
    if os.path.exists(test_image):
        # 定义候选标签
        labels = ["a cat", "a dog", "a car", "a tree", "a person", "a phone"]
        
        # 执行分类
        result = zero_shot_classification(test_image, labels)
        
        # 提取概率列表（按labels顺序）
        probs = [result["all_probs"][label] for label in labels]
        
        # 转换为numpy数组
        probs = np.array(probs)
        
        # 保存分类结果到JSON
        save_classification_result(test_image, labels, probs, clip_output_dir)
        
        # 创建可视化对比图
        create_visualization_comparison(
            test_image, 
            labels, 
            probs, 
            os.path.join(clip_output_dir, "classification_comparison.png")
        )
    else:
        print(f"测试图像不存在: {test_image}")

    # -------------------------------------------------------------------------
    # 演示2：图文检索
    # -------------------------------------------------------------------------
    print("\n2. 图文检索:")
    
    # 检查图像目录是否存在且非空
    if os.path.exists(DATA_IMAGES) and len(os.listdir(DATA_IMAGES)) > 0:
        # 执行检索，查询"a person"
        retrieval_result = text_image_retrieval("a person", DATA_IMAGES, top_k=3)
        
        if retrieval_result:
            # 保存检索结果到JSON
            save_retrieval_result(
                retrieval_result["query"], 
                retrieval_result["results"], 
                clip_output_dir
            )
            
            # 创建检索结果对比图
            create_retrieval_comparison(
                retrieval_result["query"], 
                retrieval_result["results"], 
                DATA_IMAGES, 
                os.path.join(clip_output_dir, "retrieval_comparison.png")
            )

    # -------------------------------------------------------------------------
    # 完成提示
    # -------------------------------------------------------------------------
    print(f"\n所有结果已保存至: {clip_output_dir}")