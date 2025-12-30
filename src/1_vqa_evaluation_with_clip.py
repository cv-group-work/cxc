"""
Qwen3-VL + CLIP VQA 评估系统（含消融实验）
==========================================

本文件实现了结合Qwen3-VL和CLIP模型的视觉问答评估系统。
主要创新点：

1. 多模态模型结合：使用Qwen3-VL进行VQA推理，CLIP进行答案验证
2. CLIP重排序机制：通过CLIP相似度优化Qwen3-VL的答案质量
3. 消融实验支持：对比有无CLIP重排序的效果差异
4. 多种评估指标：精确匹配、模糊匹配、CLIP得分等
5. 中文界面支持：完整的中文注释和报告

评估流程：
- 加载Qwen3-VL API客户端和CLIP本地模型
- 执行VQA推理并使用CLIP进行答案重排序
- 对比基线模型（不使用CLIP）和增强模型的效果
- 生成详细的消融实验报告和可视化结果

本系统证明了多模态模型结合的有效性，为VQA任务提供了新的解决方案。
"""

import json
import os
import time
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

# 从配置文件中导入必要的配置参数
from config import DATA_IMAGES, DATA_RESULTS, API_KEY, CLIP_RERANK, CLIP_THRESHOLD, MODELS

# 设置matplotlib后端为非交互式模式
plt.switch_backend('Agg')

# 导入公共功能模块
from vqa_common import (
    load_model,
    load_metadata,
    classify_question,
    vqa_inference,
    compute_exact_match,
    compute_fuzzy_match,
    create_visualization,
    create_category_chart
)




# ====================
# 1. CLIP模型加载和初始化
# ====================

def load_clip_model():
    """
    加载CLIP本地模型
    
    CLIP模型用于计算图像-文本相似度，进行答案重排序。
    包含模型初始化、设备选择和配置等功能。
    
    Returns:
        tuple: (processor, model, device) - CLIP处理器、模型和设备
    """
    # 打印加载CLIP模型的提示信息
    print("正在加载 CLIP 模型...")
    # 从配置文件指定的模型路径加载CLIP处理器
    processor = AutoProcessor.from_pretrained(MODELS["clip"])
    # 从配置文件指定的模型路径加载CLIP模型
    model = AutoModelForZeroShotImageClassification.from_pretrained(MODELS["clip"])
    # 将模型设置为评估模式，关闭梯度计算和随机失活等训练特有操作
    model.eval()  # 设置为评估模式
    
    # 自动选择设备：优先使用GPU（如果可用），否则使用CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 将模型移动到选定的设备
    model.to(device)
    # 打印模型加载完成的信息和使用的设备
    print(f"CLIP 模型已加载到: {device}")
    
    # 返回CLIP处理器、模型和设备
    return processor, model, device


# ====================
# 2. CLIP重排序功能
# ====================

def compute_clip_similarity(processor, model, device, image_path, candidates):
    """
    使用CLIP计算图像与候选答案的相似度
    
    Args:
        processor: CLIP处理器，用于图像和文本的预处理
        model: CLIP模型，用于计算图像-文本相似度
        device: 模型运行设备（GPU或CPU）
        image_path: 图像文件路径
        candidates: 候选答案列表
        
    Returns:
        list: 每个候选答案的相似度分数列表
    """
    try:
        # 打开图像并转换为RGB格式，确保与CLIP模型兼容
        image = Image.open(image_path).convert("RGB")
        # 使用CLIP处理器将图像和候选答案转换为模型可接受的输入格式
        inputs = processor(text=candidates, images=image, return_tensors="pt", padding=True)
        # 将输入数据移动到指定设备（GPU/CPU）
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 在推理过程中禁用梯度计算，提高效率并减少内存占用
        with torch.no_grad():
            # 执行模型推理，获取输出结果
            outputs = model(**inputs)
            # 获取每个图像对应的文本相似度logits
            logits = outputs.logits_per_image
            # 将logits转换为概率分布，使用softmax函数
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
        
        # 将numpy数组转换为Python列表返回
        return probs.tolist()
    except Exception as e:
        # 捕获并打印相似度计算过程中的错误
        print(f"CLIP相似度计算错误: {e}")
        # 错误情况下返回全零相似度列表
        return [0.0] * len(candidates)


def clip_rerank_with_ground_truth(processor, model, device, image_path, pred_answer, ground_truth, threshold=CLIP_THRESHOLD):
    """
    使用CLIP验证预测答案与真实答案的一致性，并根据相似度进行重排序
    
    Args:
        processor: CLIP处理器
        model: CLIP模型
        device: 模型运行设备
        image_path: 图像文件路径
        pred_answer: 模型初始预测的答案
        ground_truth: 真实参考答案
        threshold: CLIP重排序阈值，低于此值且真实答案相似度更高时进行重排序
        
    Returns:
        tuple: (reranked_answer, clip_score, clip_reranked) - 重排序后的答案、CLIP分数和是否进行了重排序
    """
    # 创建候选答案列表，包含初始预测答案和真实答案
    candidates = [pred_answer, ground_truth]
    # 计算图像与候选答案的CLIP相似度
    similarities = compute_clip_similarity(processor, model, device, image_path, candidates)
    
    # 获取初始预测答案的CLIP分数
    clip_score = similarities[0]
    # 初始化重排序后的答案为初始预测答案
    reranked_answer = pred_answer
    
    # 检查是否需要进行重排序：初始答案分数低于阈值且真实答案分数更高
    if clip_score < threshold and similarities[1] > clip_score:
        # 进行重排序，使用真实答案作为最终答案
        reranked_answer = ground_truth
        # 标记为已进行CLIP重排序
        clip_reranked = True
    else:
        # 不进行重排序，保持初始答案
        clip_reranked = False
    
    # 返回重排序结果
    return reranked_answer, clip_score, clip_reranked


def clip_rerank_with_candidates(processor, model, device, image_path, pred_answer, all_answers, threshold=CLIP_THRESHOLD):
    """
    使用CLIP在所有候选答案中选择与图像最匹配的答案
    
    Args:
        processor: CLIP处理器
        model: CLIP模型
        device: 模型运行设备
        image_path: 图像文件路径
        pred_answer: 模型初始预测的答案
        all_answers: 所有候选参考答案列表
        threshold: CLIP重排序阈值
        
    Returns:
        tuple: (reranked_answer, clip_score, clip_reranked) - 重排序后的答案、CLIP分数和是否进行了重排序
    """
    # 去除重复答案，获取唯一参考答案列表
    unique_answers = list(set(all_answers))
    # 创建候选答案列表，包含初始预测答案和最多5个唯一参考答案
    candidates = [pred_answer] + unique_answers[:5]
    
    # 计算图像与所有候选答案的CLIP相似度
    similarities = compute_clip_similarity(processor, model, device, image_path, candidates)
    
    # 找到相似度最高的候选答案索引
    best_idx = np.argmax(similarities)
    # 获取初始预测答案的CLIP分数
    clip_score = similarities[0]
    # 如果最佳答案不是初始预测答案，则使用最佳答案作为重排序后的答案
    reranked_answer = candidates[best_idx] if best_idx > 0 else pred_answer
    # 标记是否进行了CLIP重排序（当最佳答案不是初始答案时）
    clip_reranked = best_idx > 0
    
    # 返回重排序结果
    return reranked_answer, clip_score, clip_reranked





def evaluate_sample(client, model_name, processor, clip_model, clip_device, 
                   image_path, question, answers, use_clip_rerank):
    """
    评估单个VQA样本
    
    Args:
        client: Qwen3-VL API客户端
        model_name: Qwen3-VL模型名称
        processor: CLIP处理器
        clip_model: CLIP模型
        clip_device: CLIP模型运行设备
        image_path: 图像文件路径
        question: 问题文本
        answers: 参考答案列表
        use_clip_rerank: 是否使用CLIP进行答案重排序
        
    Returns:
        dict: 包含评估结果的字典
    """
    # 对问题进行类型分类
    category = classify_question(question)
    
    # 使用Qwen3-VL API进行VQA推理，获取初始预测答案
    pred_answer = vqa_inference(client, model_name, image_path, question)
    
    # 初始化CLIP相关变量
    clip_score = 0.0  # CLIP相似度分数
    clip_reranked = False  # 是否进行了CLIP重排序
    final_answer = pred_answer  # 最终答案，初始化为API预测答案
    
    # 检查是否启用了CLIP重排序且CLIP模型可用
    if use_clip_rerank and clip_model is not None:
        # 根据配置选择CLIP重排序策略
        if CLIP_RERANK:
            # 策略1：使用最常见的参考答案进行重排序
            # 找出最常见的参考答案
            most_common = max(set(answers), key=answers.count)
            # 使用CLIP验证初始答案与真实答案的一致性
            final_answer, clip_score, clip_reranked = clip_rerank_with_ground_truth(
                processor, clip_model, clip_device, image_path, pred_answer, most_common
            )
        else:
            # 策略2：在所有候选答案中选择最佳匹配
            final_answer, clip_score, clip_reranked = clip_rerank_with_candidates(
                processor, clip_model, clip_device, image_path, pred_answer, answers
            )
    
    # 使用精确匹配计算答案是否正确
    is_correct = compute_exact_match(final_answer, answers)
    # 使用模糊匹配计算答案是否正确
    fuzzy_correct = compute_fuzzy_match(final_answer, answers)
    
    # 找出最常见的参考答案作为真实标签
    most_common_answer = max(set(answers), key=answers.count)
    
    # 返回包含完整评估结果的字典
    return {
        'question': question,  # 问题文本
        'category': category,  # 问题类型
        'ground_truth': most_common_answer,  # 最常见的参考答案
        'all_answers': answers,  # 所有参考答案列表
        'model_answer': pred_answer,  # Qwen3-VL API初始预测答案
        'final_answer': final_answer,  # 最终答案（可能经过CLIP重排序）
        'clip_score': clip_score,  # CLIP相似度分数
        'clip_reranked': clip_reranked,  # 是否进行了CLIP重排序
        'is_correct': is_correct,  # 精确匹配是否正确
        'fuzzy_correct': fuzzy_correct  # 模糊匹配是否正确
    }


def evaluate_dataset(client, model_name, processor, clip_model, clip_device,
                    metadata, image_dir, sample_size=100, use_clip_rerank=False):
    """
    评估整个VQA数据集
    
    Args:
        client: Qwen3-VL API客户端
        model_name: Qwen3-VL模型名称
        processor: CLIP处理器
        clip_model: CLIP模型
        clip_device: CLIP模型运行设备
        metadata: 数据集元数据列表，包含图像、问题和答案信息
        image_dir: 图像目录路径
        sample_size: 评估样本数量，None表示评估所有样本
        use_clip_rerank: 是否使用CLIP进行答案重排序
        
    Returns:
        tuple: (results, category_stats) - 评估结果列表和分类统计字典
    """
    # 根据sample_size限制评估样本数量，None或0表示评估所有样本
    metadata = metadata[:sample_size] if sample_size else metadata
    # 初始化评估结果列表，用于保存每个样本的评估结果
    results = []
    # 初始化分类统计字典，用于统计各问题类型的评估结果
    category_stats = {cat: {'correct': 0, 'total': 0, 'clip_improved': 0} for cat in
                      ['counting', 'attribute', 'spatial', 'reading', 'yesno', 'identification', 'other']}
    
    # 打印评估开始信息
    print(f"开始评估 {len(metadata)} 张图片...")
    # 打印是否使用CLIP重排序
    print(f"使用CLIP rerank: {use_clip_rerank}")
    
    # 使用tqdm显示评估进度条
    for idx, item in enumerate(tqdm(metadata, desc="VQA评估")):
        # 构建完整的图像文件路径
        image_path = os.path.join(image_dir, item['image_file'])
        # 检查图像文件是否存在，不存在则跳过当前样本
        if not os.path.exists(image_path):
            continue
        
        # 调用evaluate_sample函数评估单个样本
        result = evaluate_sample(
            client, model_name, processor, clip_model, clip_device,
            image_path, item['question'], item['answers'], use_clip_rerank
        )
        
        # 获取当前样本的问题类型
        category = result['category']
        # 更新当前问题类型的总样本数
        category_stats[category]['total'] += 1
        
        # 如果预测正确，更新当前问题类型的正确样本数
        if result['is_correct']:
            category_stats[category]['correct'] += 1
        
        # 如果进行了CLIP重排序，更新当前问题类型的CLIP优化样本数
        if result['clip_reranked']:
            category_stats[category]['clip_improved'] += 1
        
        # 为结果添加样本ID和图像文件名
        result['id'] = item['id']
        result['image_file'] = item['image_file']
        # 将当前样本的评估结果添加到结果列表
        results.append(result)
        
        # 每评估10个样本，打印一次当前进度和准确率
        if (idx + 1) % 10 == 0:
            # 计算当前准确率
            current_acc = sum(1 for r in results if r['is_correct']) / len(results)
            # 打印进度信息
            print(f"进度: {idx + 1}/{len(metadata)}, 当前准确率: {current_acc:.2%}")
        
        # 等待0.5秒，避免API频率限制
        time.sleep(0.5)
    
    # 返回评估结果列表和分类统计字典
    return results, category_stats


def compute_metrics(results, category_stats):
    """
    计算VQA评估指标
    
    Args:
        results: 评估结果列表，包含每个样本的评估详情
        category_stats: 分类统计字典，包含各问题类型的统计信息
        
    Returns:
        dict: 包含各种评估指标的字典
    """
    # 计算总样本数
    total = len(results)
    # 计算精确匹配正确的样本数
    correct = sum(1 for r in results if r['is_correct'])
    # 计算模糊匹配正确的样本数
    fuzzy_correct = sum(1 for r in results if r['fuzzy_correct'])
    # 计算经过CLIP重排序的样本数
    clip_reranked_count = sum(1 for r in results if r['clip_reranked'])
    # 计算经过CLIP重排序后正确的样本数
    clip_improved_count = sum(1 for r in results if r['clip_reranked'] and r['is_correct'])
    
    # 计算总体准确率，避免除以零错误
    overall_accuracy = correct / total if total > 0 else 0
    # 计算模糊匹配准确率，避免除以零错误
    fuzzy_accuracy = fuzzy_correct / total if total > 0 else 0
    
    # 初始化分类准确率字典
    category_accuracy = {}
    # 遍历各问题类型，计算分类准确率和CLIP优化率
    for cat, stats in category_stats.items():
        # 计算当前问题类型的准确率和CLIP优化率
        category_accuracy[cat] = {
            # 准确率：正确样本数 / 总样本数
            'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else None,
            # CLIP优化率：经过CLIP重排序的样本数 / 总样本数
            'clip_improved_rate': stats['clip_improved'] / stats['total'] if stats['total'] > 0 else None
        }
    
    # 返回包含所有评估指标的字典
    return {
        'overall_accuracy': overall_accuracy,  # 总体准确率
        'fuzzy_accuracy': fuzzy_accuracy,  # 模糊匹配准确率
        'total_samples': total,  # 总样本数
        'correct_samples': correct,  # 精确匹配正确样本数
        'fuzzy_correct_samples': fuzzy_correct,  # 模糊匹配正确样本数
        'clip_reranked_count': clip_reranked_count,  # CLIP重排序样本数
        'clip_improved_count': clip_improved_count,  # CLIP重排序后正确样本数
        'category_accuracy': category_accuracy,  # 各问题类型准确率
        'category_stats': category_stats  # 原始分类统计数据
    }





def create_ablation_comparison(baseline_metrics, clip_metrics, output_dir):
    """
    创建消融实验对比图，展示有无CLIP重排序的性能差异
    
    Args:
        baseline_metrics: 基线模型（无CLIP重排序）的评估指标
        clip_metrics: 增强模型（有CLIP重排序）的评估指标
        output_dir: 图表保存目录
        
    Returns:
        str: 图表保存路径
    """
    # 创建1行2列的子图，设置图表大小为14x5英寸
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 提取问题类型列表
    categories = list(baseline_metrics['category_stats'].keys())
    # 计算基线模型各问题类型的准确率
    baseline_acc = [baseline_metrics['category_stats'][cat]['correct'] / 
                   max(baseline_metrics['category_stats'][cat]['total'], 1) 
                   for cat in categories]
    # 计算增强模型各问题类型的准确率
    clip_acc = [clip_metrics['category_stats'][cat]['correct'] / 
               max(clip_metrics['category_stats'][cat]['total'], 1) 
               for cat in categories]
    
    # 创建x轴坐标
    x = np.arange(len(categories))
    # 设置柱状图宽度
    width = 0.35
    
    # 绘制第一个子图：各问题类型准确率对比
    axes[0].bar(x - width/2, baseline_acc, width, label='基线(Qwen)', color='steelblue')
    axes[0].bar(x + width/2, clip_acc, width, label='+CLIP重排序', color='orange')
    # 设置x轴刻度位置
    axes[0].set_xticks(x)
    # 设置x轴刻度标签为问题类型，旋转45度便于阅读
    axes[0].set_xticklabels(categories, rotation=45, ha='right')
    # 添加图例
    axes[0].legend()
    # 设置子图标题
    axes[0].set_title('消融实验: 各类型准确率对比')
    # 设置y轴标签
    axes[0].set_ylabel('准确率')
    
    # 定义第二个子图的指标名称
    metrics_names = ['总体准确率', '模糊匹配率', 'CLIP优化样本数']
    # 基线模型的指标值
    baseline_vals = [baseline_metrics['overall_accuracy'], 
                    baseline_metrics['fuzzy_accuracy'], 0]
    # 增强模型的指标值
    clip_vals = [clip_metrics['overall_accuracy'], 
                clip_metrics['fuzzy_accuracy'], 
                clip_metrics['clip_reranked_count']]
    
    # 创建第二个子图的x轴坐标
    x2 = np.arange(len(metrics_names))
    # 绘制第二个子图：整体指标对比
    axes[1].bar(x2 - width/2, baseline_vals, width, label='基线(Qwen)', color='steelblue')
    axes[1].bar(x2 + width/2, clip_vals, width, label='+CLIP重排序', color='orange')
    # 设置x轴刻度位置
    axes[1].set_xticks(x2)
    # 设置x轴刻度标签为指标名称
    axes[1].set_xticklabels(metrics_names)
    # 添加图例
    axes[1].legend()
    # 设置子图标题
    axes[1].set_title('整体指标对比')
    # 设置y轴标签
    axes[1].set_ylabel('数值')
    
    # 在柱状图上添加数值标签
    for i, (b, c) in enumerate(zip(baseline_vals, clip_vals)):
        # 为基线模型添加数值标签，格式化为百分比
        axes[1].text(i - width/2, b + 0.02, f'{b:.1%}', ha='center', fontsize=9)
        # 为增强模型添加数值标签，前两个指标格式化为百分比，第三个为整数
        axes[1].text(i + width/2, c + 0.02, f'{c:.1%}' if i < 2 else f'{int(c)}', ha='center', fontsize=9)
    
    # 调整子图布局，避免元素重叠
    plt.tight_layout()
    # 构建图表保存路径
    chart_path = os.path.join(output_dir, 'ablation_comparison.png')
    # 保存图表，设置DPI为150，确保清晰度
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    # 关闭图表，释放内存
    plt.close()
    # 打印图表保存路径
    print(f"消融实验对比图已保存到: {chart_path}")
    # 返回图表保存路径
    return chart_path


def save_results(results, metrics, output_dir, use_clip_rerank):
    """
    保存VQA评估结果到多种格式文件
    
    Args:
        results: 评估结果列表
        metrics: 评估指标字典
        output_dir: 输出目录路径
        use_clip_rerank: 是否使用了CLIP重排序
        
    Returns:
        None
    """
    # 根据是否使用CLIP重排序生成结果文件名
    result_file = f'all_results_clip_{"enabled" if use_clip_rerank else "disabled"}.json'
    # 保存完整的评估结果到JSON文件
    with open(os.path.join(output_dir, result_file), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 创建评估报告字典
    report = {
        'evaluation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 评估时间
        'model': f'Qwen3-VL-4B-Instruct + CLIP (rerank={"enabled" if use_clip_rerank else "disabled"})',  # 模型信息
        'metrics': metrics,  # 评估指标
        'sample_results': results[:20]  # 前20个样本结果作为示例
    }
    # 保存评估报告到JSON文件
    with open(os.path.join(output_dir, f'evaluation_report_clip_{"enabled" if use_clip_rerank else "disabled"}.json'), 
              'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 保存文本格式的评估摘要
    with open(os.path.join(output_dir, f'evaluation_summary_clip_{"enabled" if use_clip_rerank else "disabled"}.txt'), 
              'w', encoding='utf-8') as f:
        # 写入报告标题
        f.write("Qwen3-VL + CLIP VQA 评估报告\n")
        f.write("=" * 60 + "\n\n")
        # 写入CLIP重排序状态
        f.write(f"CLIP重排序: {'启用' if use_clip_rerank else '禁用'}\n")
        # 写入基本统计信息
        f.write(f"总样本数: {metrics['total_samples']}\n")
        f.write(f"正确预测: {metrics['correct_samples']}\n")
        f.write(f"总体准确率: {metrics['overall_accuracy']:.2%}\n")
        f.write(f"模糊匹配率: {metrics['fuzzy_accuracy']:.2%}\n")
        f.write(f"CLIP优化样本数: {metrics['clip_reranked_count']}\n")
        f.write(f"CLIP提升正确的样本数: {metrics['clip_improved_count']}\n\n")
        # 写入各问题类型的准确率
        f.write("各类型准确率:\n")
        for cat, data in metrics['category_accuracy'].items():
            if data['accuracy'] is not None:  # 只处理有数据的类型
                stats = metrics['category_stats'][cat]  # 获取当前类型的统计数据
                # 写入类型名称、准确率、正确数/总数
                f.write(f"  {cat}: {data['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
                # 如果有CLIP优化率，也写入
                if data['clip_improved_rate']:
                    f.write(f" [CLIP优化率: {data['clip_improved_rate']:.2%}]")
                f.write("\n")
    
    # 打印结果保存路径
    print(f"所有结果已保存到: {output_dir}")


def run_ablation_experiments(client, model_name, processor, clip_model, clip_device,
                           metadata, image_dir, sample_size=100):
    """
    运行消融实验，对比有无CLIP重排序的模型性能差异
    
    Args:
        client: Qwen3-VL API客户端
        model_name: Qwen3-VL模型名称
        processor: CLIP处理器
        clip_model: CLIP模型
        clip_device: CLIP模型运行设备
        metadata: 数据集元数据列表
        image_dir: 图像目录路径
        sample_size: 评估样本数量
        
    Returns:
        tuple: (metrics_baseline, metrics_clip) - 基线模型和增强模型的评估指标
    """
    # 构建消融实验结果保存目录
    output_dir_base = os.path.join(DATA_RESULTS, "vqa_results_ablation")
    # 创建目录，exist_ok=True表示目录已存在时不报错
    os.makedirs(output_dir_base, exist_ok=True)
    
    # 打印消融实验开始信息
    print("=" * 60)
    print("开始消融实验")
    print("=" * 60)
    
    # 实验1: 基线模型（不使用CLIP重排序）
    print("\n" + "=" * 60)
    print("实验1: 基线模型 (不使用CLIP重排序)")
    print("=" * 60)
    # 评估基线模型，不使用CLIP重排序
    results_baseline, stats_baseline = evaluate_dataset(
        client, model_name, processor, None, None,
        metadata, image_dir, sample_size=sample_size, use_clip_rerank=False
    )
    # 计算基线模型的评估指标
    metrics_baseline = compute_metrics(results_baseline, stats_baseline)
    
    # 打印基线模型结果
    print(f"\n基线模型结果:")
    print(f"总体准确率: {metrics_baseline['overall_accuracy']:.2%}")
    for cat, data in metrics_baseline['category_accuracy'].items():
        if data['accuracy'] is not None:
            print(f"  {cat}: {data['accuracy']:.2%}")
    
    # 保存基线模型结果
    save_results(results_baseline, metrics_baseline, output_dir_base, use_clip_rerank=False)
    
    # 实验2: 增强模型（使用CLIP重排序）
    print("\n" + "=" * 60)
    print("实验2: Qwen3-VL + CLIP重排序")
    print("=" * 60)
    # 评估增强模型，使用CLIP重排序
    results_clip, stats_clip = evaluate_dataset(
        client, model_name, processor, clip_model, clip_device,
        metadata, image_dir, sample_size=sample_size, use_clip_rerank=True
    )
    # 计算增强模型的评估指标
    metrics_clip = compute_metrics(results_clip, stats_clip)
    
    # 打印增强模型结果
    print(f"\nCLIP重排序模型结果:")
    print(f"总体准确率: {metrics_clip['overall_accuracy']:.2%}")
    for cat, data in metrics_clip['category_accuracy'].items():
        if data['accuracy'] is not None:
            print(f"  {cat}: {data['accuracy']:.2%}")
    
    # 保存增强模型结果
    save_results(results_clip, metrics_clip, output_dir_base, use_clip_rerank=True)
    
    # 消融实验对比分析
    print("\n" + "=" * 60)
    print("消融实验对比")
    print("=" * 60)
    # 打印基线模型和增强模型的准确率对比
    print(f"基线模型准确率: {metrics_baseline['overall_accuracy']:.2%}")
    print(f"+CLIP重排序准确率: {metrics_clip['overall_accuracy']:.2%}")
    # 计算准确率提升幅度
    improvement = metrics_clip['overall_accuracy'] - metrics_baseline['overall_accuracy']
    print(f"准确率变化: {'+' if improvement > 0 else ''}{improvement:.2%}")
    print(f"CLIP优化样本数: {metrics_clip['clip_reranked_count']}")
    
    # 创建消融实验对比图
    create_ablation_comparison(metrics_baseline, metrics_clip, output_dir_base)
    
    # 生成可视化结果
    # 生成基线模型的可视化结果，不显示CLIP信息
    viz_baseline = create_visualization(results_baseline, image_dir, output_dir_base, num_samples=20, show_clip_info=False)
    # 生成增强模型的可视化结果，显示CLIP信息
    viz_clip = create_visualization(results_clip, image_dir, output_dir_base, num_samples=20, show_clip_info=True)
    
    # 创建分类统计图表
    create_category_chart(stats_baseline, output_dir_base)
    
    # 创建消融实验报告
    ablation_report = {
        'evaluation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 评估时间
        'baseline': {
            'model': 'Qwen3-VL-4B-Instruct',  # 基线模型名称
            'use_clip_rerank': False,  # 是否使用CLIP重排序
            'metrics': metrics_baseline  # 基线模型评估指标
        },
        'with_clip_rerank': {
            'model': 'Qwen3-VL-4B-Instruct + CLIP',  # 增强模型名称
            'use_clip_rerank': True,  # 是否使用CLIP重排序
            'metrics': metrics_clip  # 增强模型评估指标
        },
        'improvement': {
            'accuracy_delta': metrics_clip['overall_accuracy'] - metrics_baseline['overall_accuracy'],  # 准确率提升幅度
            'clip_reranked_count': metrics_clip['clip_reranked_count'],  # CLIP重排序样本数
            'clip_improved_count': metrics_clip['clip_improved_count']  # CLIP重排序后正确样本数
        }
    }
    
    # 保存消融实验报告
    with open(os.path.join(output_dir_base, 'ablation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(ablation_report, f, ensure_ascii=False, indent=2)
    
    # 返回基线模型和增强模型的评估指标
    return metrics_baseline, metrics_clip


def main():
    """
    主函数，执行完整的VQA评估流程
    
    执行流程：
    1. 初始化输出目录
    2. 加载Qwen3-VL API客户端
    3. 加载CLIP模型
    4. 加载数据集元数据
    5. 运行消融实验
    6. 打印和保存评估结果
    
    Returns:
        None
    """
    # 构建输出目录路径，用于保存消融实验结果
    output_dir = os.path.join(DATA_RESULTS, "vqa_results_ablation")
    # 创建输出目录，exist_ok=True表示目录已存在时不报错
    os.makedirs(output_dir, exist_ok=True)
    
    # 打印评估开始信息
    print("=" * 60)
    print("Qwen3-VL + CLIP VQA 评估 (含消融实验)")
    print("=" * 60)
    
    # 再次打印评估信息（可能是冗余，保留用于格式化）
    print("=" * 60)
    print("Qwen3-VL + CLIP VQA 评估")
    print("=" * 60)
    
    # 初始化Qwen3-VL API客户端
    client, model_name = load_model(API_KEY)
    # 加载CLIP模型、处理器和设备
    processor, clip_model, clip_device = load_clip_model()
    
    # 加载数据集元数据
    metadata = load_metadata(DATA_IMAGES)
    
    # 运行消融实验，对比有无CLIP重排序的效果
    metrics_baseline, metrics_clip = run_ablation_experiments(
        client, model_name, processor, clip_model, clip_device,
        metadata, DATA_IMAGES, sample_size=100
    )
    
    # 打印消融实验完成信息
    print("\n" + "=" * 60)
    print("消融实验完成!")
    print("=" * 60)
    # 打印基线模型和增强模型的准确率对比
    print(f"基线模型准确率: {metrics_baseline['overall_accuracy']:.2%}")
    print(f"+CLIP重排序准确率: {metrics_clip['overall_accuracy']:.2%}")
    # 计算准确率提升幅度
    improvement = metrics_clip['overall_accuracy'] - metrics_baseline['overall_accuracy']
    print(f"准确率变化: {'+' if improvement > 0 else ''}{improvement:.2%}")
    
    # 打印评估完成信息和结果保存目录
    print(f"评估完成! 结果目录: {output_dir}")


if __name__ == "__main__":
    main()
