"""
Qwen3-VL模型复现演示
===================

本文件演示了如何使用Qwen3-VL-4B-Instruct模型进行视觉问答任务。
Qwen3-VL是阿里云通义千问系列的视觉语言模型，能够理解和回答关于图像的问题。

主要功能：
- 加载预训练的Qwen3-VL模型
- 处理图像输入和文本问题
- 生成准确的视觉问答回答

这个脚本展示了Qwen3-VL模型的基本用法，为后续的评估工作奠定基础。
"""

import torch
from PIL import Image
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from config import DATA_IMAGES, DATA_RESULTS

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans CN']
current_font = None
for font in chinese_fonts:
    if font in available_fonts:
        current_font = font
        break

# ====================
# 模型和处理器加载
# ====================

# 加载Qwen3-VL-4B-Instruct预训练模型
# torch.float16使用半精度浮点数，可以减少内存使用并提高推理速度
# device_map="cuda"自动将模型分配到GPU设备上
print("正在加载Qwen3-VL模型...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",    # 模型标识符
    dtype=torch.float16,             # 使用半精度浮点数
    device_map="cuda"               # 自动映射到CUDA设备
)

# 加载模型的处理器
# 处理器负责将原始数据转换为模型可以理解的格式
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

# ====================
# 图像加载和预处理
# ====================

# 构建测试图像的完整路径
image_path = DATA_IMAGES / "5.jpg"

# 加载图像并转换为RGB格式
# 转换RGB确保模型能够正确处理图像（避免RGBA等格式的兼容性问题）
image = Image.open(image_path).convert("RGB")

# ====================
# 构建对话消息
# ====================

# 定义用户输入消息
# Qwen3-VL使用特定的对话格式，需要包含角色信息和多模态内容
messages = [
    {
        "role": "user",  # 用户角色
        "content": [
            # 图像输入类型标识
            {"type": "image"},
            # 文本输入，包含要询问的问题
            {"type": "text", "text": "这个图片是什么?"}
        ]
    }
]

# ====================
# 输入预处理
# ====================

# 使用processor处理所有输入数据
# apply_chat_template：将对话消息转换为模型输入格式
# tokenize=False：返回未分词的字符串格式
# add_generation_prompt=True：添加生成提示，帮助模型识别需要生成回答
text = processor.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

# 使用processor处理文本和图像输入
# text=[text]：需要是列表格式，因为processor设计用于批量处理
# images=[image]：要分析的图像
# padding=True：对批次中的样本进行padding以保持长度一致
# return_tensors="pt"：返回PyTorch张量格式
inputs = processor(
    text=[text],        # 处理后的对话文本
    images=[image],     # 输入图像
    padding=True,       # 启用padding
    return_tensors="pt" # 返回PyTorch张量
).to("cuda")           # 将输入移动到GPU设备

# ====================
# 模型推理和回答生成
# ====================

# 禁用梯度计算，提高推理效率
with torch.no_grad():
    # 使用模型生成回答
    # max_new_tokens：限制生成回答的最大长度
    # temperature：控制回答的随机性，较低值使回答更确定性
    # do_sample=True：启用采样，增加回答的多样性
    outputs = model.generate(
        **inputs,                    # 解包输入参数
        max_new_tokens=512,         # 最大生成512个新token
        temperature=0.7,            # 适度的随机性
        do_sample=True              # 启用采样
    )

# ====================
# 输出后处理
# ====================

# 解码模型生成的token序列为可读的文本
# skip_special_tokens=True：跳过特殊的token（如填充符等）
answer = processor.decode(outputs[0], skip_special_tokens=True)

# ====================
# 结果保存和可视化功能
# ====================

def create_vqa_visualization_comparison(image_path, question, answer, output_path):
    """
    创建原图与VQA输出结果的对比可视化图像
    
    Args:
        image_path: 原始图像路径
        question: 输入的问题
        answer: 模型生成的回答
        output_path: 输出图像保存路径
    """
    original_image = Image.open(image_path).convert("RGB")
    
    width, height = original_image.size
    dpi = 100
    fig_width = (width * 2) / dpi
    fig_height = height / dpi
    
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    font_prop = None
    if current_font:
        font_prop = matplotlib.font_manager.FontProperties(family=current_font)
    
    display_question = question if font_prop else "Question: image content?"
    display_answer = answer if font_prop else "Answer: [Chinese text]"
    
    info_text = f"Question: {display_question}\n\nAnswer: {display_answer}"
    
    axes[1].text(0.05, 0.95, info_text, 
                 transform=axes[1].transAxes, fontsize=11,
                 verticalalignment='top', fontproperties=font_prop,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    axes[1].set_title("Qwen3-VL VQA Result", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"可视化对比图已保存至: {output_path}")


def save_vqa_result(image_path, question, answer, output_dir):
    """保存VQA结果到JSON文件"""
    result = {
        "image": str(image_path),
        "question": question,
        "answer": answer
    }
    
    output_path = os.path.join(output_dir, "vqa_result.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"VQA结果已保存至: {output_path}")
    return output_path


# ====================
# 输出后处理
# ====================

# 打印完整的对话和回答
print("=== Qwen3-VL 视觉问答演示 ===")
print(f"图像文件: {os.path.basename(image_path)}")
print(f"问题: 这个图片是什么?")
print(f"回答: {answer}")

# ====================
# 保存结果到data/results
# ====================

# 创建Qwen演示结果保存目录
qwen_output_dir = DATA_RESULTS / "qwen_demo"
os.makedirs(qwen_output_dir, exist_ok=True)

# 保存JSON结果
save_vqa_result(image_path, "这个图片是什么?", answer, qwen_output_dir)

# 创建可视化对比图
create_vqa_visualization_comparison(
    image_path, 
    "这个图片是什么?", 
    answer,
    os.path.join(qwen_output_dir, "vqa_comparison.png")
)

print(f"\n所有结果已保存至: {qwen_output_dir}")