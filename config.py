"""
项目配置文件
================

本文件包含项目中使用的所有配置参数，包括路径设置、模型配置、数据集配置等。
通过环境变量和配置文件来管理这些参数，便于在不同环境中部署和调试。
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载.env文件中的环境变量
# override=True表示.env文件中的变量会覆盖系统中已存在的同名变量
load_dotenv(override=True)

# ====================
# 路径配置
# ====================

# 项目根目录路径 - 指向包含此配置文件的目录
ROOT = Path(__file__).parent

# 数据目录配置
DATA = ROOT / "data"                    # 主数据目录
DATA_IMAGES = DATA / "images"           # 图像数据目录
DATA_RESULTS = DATA / "results"         # 评估结果保存目录

# ====================
# 模型配置
# ====================

# 预训练模型标识符配置
MODELS = {
    "clip": "openai/clip-vit-base-patch32",    # OpenAI的CLIP模型，用于图像-文本匹配
    "qwen": "Qwen/Qwen3-VL-4B-Instruct"        # 通义千问的视觉语言模型，支持图像问答
}

# ====================
# 数据集配置
# ====================

# 用于评估的数据集配置
DATASET = {
    "TextVQA": "Multimodal-Fatima/TextVQA_validation",    # TextVQA验证集，主要包含需要读取图像中文字的问题
    "VQAv2":"Multimodal-Fatima/VQAv2_sample_test"        # VQAv2测试集的样本版本
}

# ====================
# 实验参数配置
# ====================

# 评估时使用的样本数量
# 设置为100表示从完整数据集中随机选择100个样本进行评估
SAMPLE_SIZE = 100

# 计算设备自动选择
# 如果系统中有CUDA设备（GPU），则使用cuda，否则使用cpu
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

# ====================
# 提示词模板配置
# ====================

# 不同类型问题的提示词模板
# 用于指导模型回答特定类型的问题
PROMPTS = {
    "vqa": "请回答关于这张图片的以下问题: {question}",     # 通用VQA问题模板
    "description": "请详细描述这张图片",                    # 图像描述任务
    "count": "这张图片中有多少个 {object}？",               # 计数类问题模板
    "spatial": "这张图片中 {obj1} 和 {obj2} 之间的空间关系是什么？",  # 空间关系问题
    "reading": "这张图片中能看到什么文字？"                  # 文字识别问题
}

# ====================
# CLIP重排序配置
# ====================

# 是否启用CLIP重排序功能
# True表示在使用Qwen3-VL生成答案后，使用CLIP对答案进行验证和优化
CLIP_RERANK = True

# CLIP重排序的阈值
# 当CLIP相似度分数低于此阈值时，会考虑替换为其他候选答案
CLIP_THRESHOLD = 0.3

# ====================
# API配置
# ====================

# DashScope API密钥
# 从环境变量DASHSCOPE_API_KEY中获取，用于调用Qwen3-VL的API服务
API_KEY = os.getenv("DASHSCOPE_API_KEY")