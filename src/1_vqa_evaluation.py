"""
Qwen3-VL VQA 评估系统（API版本）
================================

本文件实现了基于Qwen3-VL模型的视觉问答(VQA)评估系统。
主要功能包括：

1. 使用Qwen3-VL API进行视觉问答推理
2. 对不同类型的问题进行分类评估
3. 计算多种评估指标（准确率、模糊匹配等）
4. 生成详细的可视化结果和分析报告
5. 支持TextVQA和VQAv2等数据集

评估流程：
- 加载测试数据集和元数据
- 对每个样本进行VQA推理
- 使用多种匹配策略评估答案质量
- 生成统计报告和可视化结果

本版本使用API调用方式，适合快速原型开发和测试。
"""

import json
import os
import re
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from openai import OpenAI
import base64
from io import BytesIO
import time

# 从配置文件中导入必要的配置参数
from config import DATA_IMAGES, DATA_RESULTS, API_KEY

# 设置matplotlib后端为非交互式模式，适合服务器环境
plt.switch_backend('Agg')


# ====================
# 1. 模型和客户端初始化
# ====================

def load_model(api_key):
    """
    初始化Qwen3-VL API客户端
    
    Args:
        api_key (str): DashScope API密钥
    
    Returns:
        tuple: (client, model_name) - API客户端和模型名称
    
    功能说明：
    - 创建OpenAI兼容的API客户端
    - 配置DashScope端点和认证
    - 设置使用的模型版本
    """
    print("初始化 Qwen3-VL API 客户端")
    # 创建OpenAI兼容的客户端，连接到DashScope服务
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    # 指定要使用的模型版本
    model_name = "qwen3-vl-8b-instruct"
    print(f"使用模型: {model_name}")
    return client, model_name


# ====================
# 2. 数据加载和预处理
# ====================

def load_metadata(data_dir):
    """
    加载数据集元数据
    
    Args:
        data_dir (str): 数据目录路径
    
    Returns:
        list: 包含图像、问题和答案的元数据列表
    
    元数据格式：
    [
        {
            "id": "图像ID",
            "image_file": "图像文件名",
            "question": "问题文本",
            "answers": ["答案1", "答案2", ...]
        },
        ...
    ]
    """
    # 读取metadata.json文件，包含所有测试样本的信息
    with open(os.path.join(data_dir, "metadata.json"), 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"加载了 {len(metadata)} 条元数据")
    return metadata


# ====================
# 3. 答案处理和匹配算法
# ====================

def normalize_answer(answer):
    """
    答案标准化处理
    
    对模型生成的答案进行清理和标准化，便于后续的匹配比较。
    处理包括：转小写、去除标点、标准化空格等。
    
    Args:
        answer (str): 原始答案文本
    
    Returns:
        str: 标准化后的答案文本
    """
    if not answer:
        return ""
    # 转换为小写并去除首尾空格
    answer = answer.lower().strip()
    # 去除标点符号，用空格替代
    answer = re.sub(r'[^\w\s]', ' ', answer)
    # 将多个连续空格替换为单个空格
    answer = re.sub(r'\s+', ' ', answer)
    return answer.strip()


def fuzzy_match(pred, target):
    """
    模糊匹配算法
    
    实现多种模糊匹配策略，提高评估的鲁棒性。
    包括精确匹配、子串匹配、单词重叠度计算等。
    
    Args:
        pred (str): 预测答案
        target (str): 目标答案
    
    Returns:
        bool: 是否匹配成功
    """
    pred_norm = normalize_answer(pred)
    target_norm = normalize_answer(target)
    
    # 1. 精确匹配
    if pred_norm == target_norm:
        return True
    
    # 2. 目标答案包含在预测答案中
    if target_norm in pred_norm:
        return True
    
    # 3. 预测答案包含在目标答案中（且预测答案长度足够）
    if pred_norm in target_norm and len(pred_norm) > 3:
        return True
    
    # 4. 单词重叠度计算
    pred_words = set(pred_norm.split())
    target_words = set(target_norm.split())
    # 如果目标答案有词汇，且重叠率超过70%
    if target_words and len(pred_words & target_words) / len(target_words) > 0.7:
        return True
    
    return False


def compute_accuracy(pred, targets):
    """
    计算预测答案的准确率
    
    使用多种匹配策略综合评估预测答案的质量。
    支持精确匹配、模糊匹配和部分匹配。
    
    Args:
        pred (str): 预测答案
        targets (list): 目标答案列表（通常有多个标注答案）
    
    Returns:
        bool: 预测是否正确
    """
    pred_norm = normalize_answer(pred)
    
    best_match = None
    best_score = 0
    
    # 遍历所有目标答案，找出最佳匹配
    for target in targets:
        target_norm = normalize_answer(target)
        
        # 精确匹配
        if pred_norm == target_norm:
            return True
        
        score = 0
        
        # 部分匹配评分
        if target_norm in pred_norm:
            score = 0.9  # 高分：目标答案完全包含在预测中
        elif pred_norm in target_norm:
            score = 0.8  # 中高分：预测答案包含在目标中
        else:
            # 单词重叠度评分
            common = set(pred_norm.split()) & set(target_norm.split())
            if target_norm.split():
                score = len(common) / len(set(target_norm.split()))
        
        # 更新最佳匹配
        if score > best_score:
            best_score = score
            best_match = target
    
    # 设置评分阈值，高于60%认为匹配成功
    return best_score >= 0.6


# ====================
# 4. 问题分类系统
# ====================

def classify_question(question):
    """
    问题类型自动分类
    
    基于关键词匹配对VQA问题进行分类，
    支持计数、属性、空间关系、文字识别等类型。
    
    Args:
        question (str): 问题文本
    
    Returns:
        str: 问题类型标签
    """
    q = question.lower()
    
    # 定义各类问题的关键词
    categories = {
        'counting': ['how many', '多少', 'count', 'number of', '数量'],
        'attribute': ['what color', 'what brand', 'what type', 'what kind', 'what year', 
                      'what time', '颜色', '品牌', '类型', '年份', '时间'],
        'spatial': ['where', 'what is on the left', 'what is on the right', 'what is in front',
                    '位置', '左边', '右边', '前面', '后面', '上面', '下面'],
        'reading': ['what does it say', 'what does the sign say', 'what does the text say',
                    'what is written', 'read', '读取', '文字', '写的', '说什么'],
        'yesno': ['is this', 'are these', 'was the', 'does her shirt say', '是否', '是不是'],
        'identification': ['who is', 'what is the name', 'what is this', 'who was',
                          '谁', '名称', '是什么']
    }
    
    # 遍历分类，检查是否包含关键词
    for cat, keywords in categories.items():
        if any(kw in q for kw in keywords):
            return cat
    return 'other'


# ====================
# 5. 图像处理工具
# ====================

def image_to_base64(image_path):
    """
    将图像转换为base64编码
    
    用于API调用时的图像传输。
    包含图像预处理：格式转换、尺寸调整、质量优化等。
    
    Args:
        image_path (str): 图像文件路径
    
    Returns:
        str: base64编码的图像字符串，失败返回None
    """
    try:
        with Image.open(image_path) as img:
            # 转换为RGB格式确保兼容性
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 图像尺寸限制：最大边不超过1024像素
            max_size = 1024
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # 转换为base64编码
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)  # 85%质量平衡大小和清晰度
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return img_str
    except Exception as e:
        print(f"图片处理错误 {image_path}: {e}")
        return None


# ====================
# 6. VQA推理引擎
# ====================

def vqa_inference(client, model_name, image_path, question, max_retries=3):
    """
    视觉问答推理函数
    
    调用Qwen3-VL API进行图像问答推理。
    包含错误处理、重试机制和响应解析。
    
    Args:
        client: API客户端
        model_name (str): 模型名称
        image_path (str): 图像路径
        question (str): 问题文本
        max_retries (int): 最大重试次数
    
    Returns:
        str: 模型回答或错误信息
    """
    for attempt in range(max_retries):
        try:
            # 图像预处理和编码
            base64_image = image_to_base64(image_path)
            if not base64_image:
                return f"错误: 无法处理图片"
            
            # 构建API消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": question + "\n请简短回答，只回答关键信息，不需要解释。"
                        }
                    ]
                }
            ]
            
            # 调用API
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=128,    # 限制回答长度
                temperature=0.1,   # 低温度确保回答稳定
                stream=False       # 禁用流式输出
            )
            
            # 解析回答
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            print(f"API调用错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # 重试前等待
            else:
                return f"错误: {str(e)}"


# ====================
# 7. 数据集评估引擎
# ====================

def evaluate_dataset(client, model_name, metadata, image_dir, sample_size=100):
    """
    评估整个数据集
    
    对数据集中的样本进行批量评估，
    包含问题分类、推理、结果统计等功能。
    
    Args:
        client: API客户端
        model_name (str): 模型名称
        metadata (list): 数据集元数据
        image_dir (str): 图像目录路径
        sample_size (int): 评估样本数量
    
    Returns:
        tuple: (results, category_stats) - 评估结果和分类统计
    """
    # 限制评估样本数量
    metadata = metadata[:sample_size] if sample_size else metadata
    results = []
    
    # 初始化各问题类型的统计字典
    category_stats = {cat: {'correct': 0, 'total': 0} for cat in
                      ['counting', 'attribute', 'spatial', 'reading', 'yesno', 'identification', 'other']}
    
    print(f"开始评估 {len(metadata)} 张图片...")
    
    # 遍历每个样本进行评估
    for idx, item in enumerate(tqdm(metadata, desc="VQA评估")):
        image_path = os.path.join(image_dir, item['image_file'])
        if not os.path.exists(image_path):
            continue  # 跳过不存在的图像
        
        # 问题分类
        category = classify_question(item['question'])
        category_stats[category]['total'] += 1
        
        # VQA推理
        pred_answer = vqa_inference(client, model_name, image_path, item['question'])
        
        # 准确率计算
        is_correct = compute_accuracy(pred_answer, item['answers'])
        
        # 更新统计
        if is_correct:
            category_stats[category]['correct'] += 1
        
        # 找出最常见的标注答案
        most_common_answer = max(set(item['answers']), key=item['answers'].count)
        
        # 保存详细结果
        results.append({
            'id': item['id'],
            'image_file': item['image_file'],
            'question': item['question'],
            'category': category,
            'ground_truth': most_common_answer,
            'all_answers': item['answers'],
            'model_answer': pred_answer,
            'is_correct': is_correct
        })
        
        # 进度报告
        if (idx + 1) % 10 == 0:
            current_acc = sum(1 for r in results if r['is_correct']) / len(results)
            print(f"进度: {idx + 1}/{len(metadata)}, 当前准确率: {current_acc:.2%}")
        
        # 避免API频率限制
        time.sleep(0.5)
    
    return results, category_stats


# ====================
# 8. 评估指标计算
# ====================

def compute_metrics(results, category_stats):
    """
    计算评估指标
    
    汇总评估结果，计算总体和各分类的准确率等指标。
    
    Args:
        results (list): 评估结果列表
        category_stats (dict): 分类统计字典
    
    Returns:
        dict: 包含各种评估指标的字典
    """
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    overall_accuracy = correct / total if total > 0 else 0
    
    # 计算各问题类型的准确率
    category_accuracy = {}
    for cat, stats in category_stats.items():
        category_accuracy[cat] = stats['correct'] / stats['total'] if stats['total'] > 0 else None
    
    return {
        'overall_accuracy': overall_accuracy,
        'total_samples': total,
        'correct_samples': correct,
        'category_accuracy': category_accuracy,
        'category_stats': category_stats
    }


def create_visualization(results, image_dir, output_dir, num_samples=20):
    """
    创建VQA评估结果的可视化展示
    
    生成包含成功和失败案例的图像网格，直观展示模型的评估表现。
    每个子图显示原始图像、问题、模型答案和评估结果。
    
    Args:
        results (list): 评估结果列表
        image_dir (str): 图像目录路径
        output_dir (str): 输出目录路径
        num_samples (int): 要展示的样本数量
    
    Returns:
        str: 可视化图像的保存路径
    """
    # 分离成功和失败的案例
    success = [r for r in results if r['is_correct']]
    failure = [r for r in results if not r['is_correct']]
    
    selected = []
    # 平衡选择成功和失败的案例
    num_success = min(num_samples // 2, len(success))
    num_failure = min(num_samples - num_success, len(failure))
    
    # 随机选择成功案例
    if num_success > 0:
        indices = np.random.choice(len(success), num_success, replace=False)
        for idx in indices:
            selected.append(success[idx])
    
    # 随机选择失败案例
    if num_failure > 0:
        indices = np.random.choice(len(failure), num_failure, replace=False)
        for idx in indices:
            selected.append(failure[idx])
    
    # 随机打乱顺序
    np.random.shuffle(selected)
    
    # 计算子图布局
    cols, rows = 4, (len(selected) + 3) // 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()
    
    # 为每个选中的案例创建子图
    for idx, result in enumerate(selected):
        if idx >= len(axes):
            break
        ax = axes[idx]
        try:
            # 加载并显示图像
            img = Image.open(os.path.join(image_dir, result['image_file']))
            ax.imshow(img)
            ax.axis('off')  # 隐藏坐标轴
            
            # 设置结果状态和颜色
            status = "[成功]" if result['is_correct'] else "[失败]"
            color = 'green' if result['is_correct'] else 'red'
            
            # 设置子图标题，包含问题、答案和评估结果
            ax.set_title(
                f"ID:{result['id']} [{result['category']}] {status}\n"
                f"问题: {result['question'][:35]}...\n"
                f"答案: {result['model_answer'][:30]}",
                fontsize=8, color=color
            )
        except Exception as e:
            # 图像加载失败的处理
            ax.text(0.5, 0.5, f"图像加载失败", ha='center', va='center')
            ax.axis('off')
    
    # 隐藏多余的子图
    for ax in axes.flat[len(selected):]:
        ax.axis('off')
    
    # 调整布局并保存
    plt.tight_layout()
    viz_path = os.path.join(output_dir, 'vqa_visualization.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化结果已保存到: {viz_path}")
    return viz_path


def create_category_chart(category_stats, output_dir):
    """
    创建问题类型统计图表
    
    生成柱状图展示各问题类型的样本数量分布和准确率表现。
    
    Args:
        category_stats (dict): 分类统计数据
        output_dir (str): 输出目录路径
    
    Returns:
        str: 图表保存路径
    """
    # 提取统计数据
    categories = list(category_stats.keys())
    totals = [stats['total'] for stats in category_stats.values()]
    correct = [stats['correct'] for stats in category_stats.values()]
    accuracies = [c / t if t > 0 else 0 for c, t in zip(correct, totals)]
    
    # 创建子图：样本统计 + 准确率
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1：样本数量统计
    x = np.arange(len(categories))
    ax1.bar(x - 0.175, totals, 0.35, label='总计', color='steelblue')
    ax1.bar(x + 0.175, correct, 0.35, label='正确', color='green')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend()
    ax1.set_title('问题类型统计')
    
    # 子图2：各类型准确率
    # 根据准确率设置颜色：绿色>50%，橙色>30%，红色<=30%
    colors = ['green' if acc > 0.5 else 'orange' if acc > 0.3 else 'red' for acc in accuracies]
    ax2.bar(categories, accuracies, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.axhline(y=0.5, color='gray', linestyle='--', label='50%基线')
    ax2.set_title('各问题类型准确率')
    ax2.legend()
    
    # 在柱状图上标注百分比
    for i, acc in enumerate(accuracies):
        if acc > 0:
            ax2.text(i, acc + 0.02, f'{acc:.1%}', ha='center', fontsize=9)
    
    # 调整布局并保存
    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'category_statistics.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"分类统计图表已保存到: {chart_path}")
    return chart_path


def save_results(results, metrics, output_dir):
    """
    保存评估结果到多种格式
    
    保存详细的评估结果、报告摘要和统计信息到文件。
    
    Args:
        results (list): 完整评估结果
        metrics (dict): 评估指标
        output_dir (str): 输出目录
    """
    # 保存完整结果为JSON格式
    with open(os.path.join(output_dir, 'all_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 创建评估报告
    report = {
        'evaluation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model': 'qwen3-vl-8b-instruct (API)',
        'metrics': metrics,
        'sample_results': results[:20]  # 只保存前20个样本作为示例
    }
    with open(os.path.join(output_dir, 'evaluation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 创建文本格式的评估摘要
    with open(os.path.join(output_dir, 'evaluation_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("Qwen3-VL VQA 评估报告 (API版本)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"总样本数: {metrics['total_samples']}\n")
        f.write(f"正确预测: {metrics['correct_samples']}\n")
        f.write(f"总体准确率: {metrics['overall_accuracy']:.2%}\n\n")
        f.write("各类型准确率:\n")
        for cat, acc in metrics['category_accuracy'].items():
            if acc is not None:
                stats = metrics['category_stats'][cat]
                f.write(f"  {cat}: {acc:.2%} ({stats['correct']}/{stats['total']})\n")
    
    print(f"所有结果已保存到: {output_dir}")


def main():
    """
    主评估函数
    
    执行完整的VQA评估流程：
    1. 初始化模型和客户端
    2. 加载数据集
    3. 执行评估
    4. 计算指标
    5. 生成可视化和报告
    6. 保存结果
    """
    # 创建输出目录
    output_dir = os.path.join(DATA_RESULTS, "vqa_results_api")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 50)
    print("Qwen3-VL VQA 评估 (API版本)")
    print("=" * 50)
    
    # 初始化模型客户端
    client, model_name = load_model(API_KEY)
    
    # 加载测试数据
    metadata = load_metadata(DATA_IMAGES)
    
    # 执行数据集评估
    results, category_stats = evaluate_dataset(client, model_name, metadata, DATA_IMAGES, sample_size=100)
    
    # 计算评估指标
    metrics = compute_metrics(results, category_stats)
    
    # 打印评估结果摘要
    print(f"\n评估结果:")
    print(f"总体准确率: {metrics['overall_accuracy']:.2%}")
    for cat, acc in metrics['category_accuracy'].items():
        if acc is not None:
            stats = metrics['category_stats'][cat]
            print(f"  {cat}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    
    # 生成可视化图表
    create_visualization(results, DATA_IMAGES, output_dir)
    create_category_chart(category_stats, output_dir)
    
    # 保存所有结果
    save_results(results, metrics, output_dir)
    
    print(f"\n评估完成! 结果目录: {output_dir}")


if __name__ == "__main__":
    main()
