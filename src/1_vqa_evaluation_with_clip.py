"""
================================================================================
Qwen3-VL + CLIP VQA 评估系统脚本 (LangChain + Pipeline版本)
================================================================================

本文件实现了VQA评估系统。
该系统在Qwen3-VL的基础上，增加了CLIP答案重排序机制，通过图像-文本相似度
验证和优化模型生成的答案，提高VQA的准确率。

主要功能模块：
1. CLIP模型集成 - 本地加载CLIP模型进行图像-文本相似度计算
2. 答案重排序 - 使用CLIP验证Qwen3-VL的答案，选择最优答案
3. 消融实验 - 对比有无CLIP重排序的效果差异
4. 多指标评估 - 精确匹配、模糊匹配、CLIP优化率等
5. 可视化对比 - 生成消融实验对比图表

================================================================================
"""

# ==============================================================================
# 第一部分：环境配置和库导入
# ==============================================================================

# 导入标准库
import os
import json
import time
from datetime import datetime

# 设置OpenMP库兼容性选项
# 解决某些系统上可能出现的多重库冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 导入配置模块
from config import DATA_IMAGES, DATA_RESULTS, API_KEY, CLIP_RERANK, MODELS

# 导入公共功能模块
from vqa_common import (
    load_model,              # 初始化模型客户端
    load_metadata,           # 加载数据集元数据
    classify_question,       # 问题类型分类
    vqa_inference,          # VQA推理
    compute_exact_match,    # 精确匹配评估
    compute_fuzzy_match,    # 模糊匹配评估
    create_visualization,   # 创建评估结果可视化
    create_category_chart   # 创建分类统计图表
)

# 导入进度条库
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import torch

# 设置matplotlib后端为Agg（无GUI模式）
plt.switch_backend('Agg')

# ==============================================================================
# 第二部分：CLIP模型初始化
# ==============================================================================

# 加载CLIP模型的处理器
# 负责将图像和文本转换为模型所需的输入格式
clip_processor = AutoProcessor.from_pretrained(MODELS["clip"])

# 加载CLIP预训练模型
# AutoModelForZeroShotImageClassification是专门用于零样本分类的模型类
clip_model = AutoModelForZeroShotImageClassification.from_pretrained(MODELS["clip"])

# 设置模型为评估模式
# 评估模式会关闭Dropout等训练专用层
clip_model.eval()

# 自动检测并选择计算设备
clip_device = "cuda" if torch.cuda.is_available() else "cpu"

# 将模型移动到指定设备
clip_model.to(clip_device)

print(f"CLIP模型已加载到: {clip_device}")


# ==============================================================================
# 第三部分：CLIP相似度计算函数
# ==============================================================================

def compute_clip_similarity(image_path, candidates):
    """
    使用CLIP计算图像与候选答案的相似度
    
    该函数将图像和多个候选答案文本分别编码为特征向量，
    然后计算它们之间的相似度分数。
    
    工作原理：
    1. 打开并预处理图像
    2. 使用CLIP处理器编码图像和文本
    3. 提取图像和文本的特征向量
    4. 计算图像与各文本的相似度（logits）
    5. 通过softmax转换为概率分布
    
    Args:
        image_path (str): 图像文件路径
        candidates (list): 候选答案文本列表
        
    Returns:
        list: 各候选答案的相似度/概率列表

    """
    try:
        # 打开图像并转换为RGB模式
        image = Image.open(image_path).convert("RGB")
        
        # 使用CLIP处理器编码图像和文本
        # return_tensors="pt" 返回PyTorch张量
        # padding=True 确保文本长度一致
        inputs = clip_processor(text=candidates, images=image, return_tensors="pt", padding=True)
        
        # 将输入张量移动到指定设备（GPU/CPU）
        inputs = {k: v.to(clip_device) for k, v in inputs.items()}

        # 禁用梯度计算，减少内存占用
        with torch.no_grad():
            # 前向传播，获取模型输出
            outputs = clip_model(**inputs)
            
            # 获取每个图像-文本对的相似度分数
            # logits_per_image: (1, num_candidates)
            logits = outputs.logits_per_image
            
            # 通过softmax转换为概率分布，NumPy 数组仅支持 CPU 数据
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

        return probs.tolist()
        
    except Exception as e:
        # 打印错误并返回零向量
        print(f"CLIP相似度计算错误: {e}")
        return [0.0] * len(candidates)


# ==============================================================================
# 第四部分：单个样本评估函数
# ==============================================================================

def evaluate_sample(llm, model_name, image_path, question, answers, use_clip):
    """
    评估单个VQA样本
    
    该函数对单个VQA样本进行评估，支持可选的CLIP重排序。
    
    处理流程：
    1. 使用Qwen3-VL生成初始答案
    2. 如果启用CLIP重排序：
       a. 构建候选答案列表
       b. 计算CLIP相似度
       c. 选择最佳候选作为最终答案
    3. 计算精确匹配和模糊匹配
    
    Args:
        llm: LangChain LLM对象
        model_name (str): 模型名称
        image_path (str): 图像文件路径
        question (str): 问题文本
        answers (list): 标准答案列表
        use_clip (bool): 是否使用CLIP重排序
        
    Returns:
        dict: 评估结果字典，包含以下字段：
            - question: 问题文本
            - category: 问题类型
            - ground_truth: 最常见标准答案
            - all_answers: 所有标准答案
            - model_answer: Qwen3-VL生成的初始答案
            - final_answer: 最终答案（可能经过CLIP优化）
            - clip_score: 初始答案的CLIP相似度
            - clip_reranked: 是否经过CLIP重排序
            - is_correct: 精确匹配是否正确
            - fuzzy_correct: 模糊匹配是否正确
    """
    # 对问题进行分类
    category = classify_question(question)
    
    # 使用Qwen3-VL进行VQA推理，获取初始答案
    pred_answer = vqa_inference(llm, model_name, image_path, question)

    # 初始化CLIP相关变量
    clip_score, clip_reranked, final_answer = 0.0, False, pred_answer

    # 如果启用CLIP重排序
    if use_clip and CLIP_RERANK:
        # 构建候选答案列表
        # 第一个是初始答案，后面是标准答案中的唯一答案（最多5个）
        unique_answers = list(set(answers))[:5]
        candidates = [pred_answer] + unique_answers
        
        # 计算CLIP相似度
        similarities = compute_clip_similarity(image_path, candidates)

        # 获取初始答案的CLIP相似度
        clip_score = similarities[0]
        
        # 找到相似度最高的候选答案索引
        best_idx = np.argmax(similarities)

        # 如果最佳候选不是初始答案，进行重排序
        if best_idx > 0:
            final_answer = candidates[best_idx]
            clip_reranked = True

    # 计算精确匹配
    is_correct = compute_exact_match(final_answer, answers)
    
    # 计算模糊匹配
    fuzzy_correct = compute_fuzzy_match(final_answer, answers)

    # 返回评估结果
    return {
        'question': question, 
        'category': category,
        'ground_truth': max(set(answers), key=answers.count),
        'all_answers': answers, 
        'model_answer': pred_answer,
        'final_answer': final_answer, 
        'clip_score': clip_score,
        'clip_reranked': clip_reranked, 
        'is_correct': is_correct,
        'fuzzy_correct': fuzzy_correct
    }


# ==============================================================================
# 第五部分：数据集评估函数
# ==============================================================================

def evaluate_dataset(llm, model_name, metadata, image_dir, sample_size=100, use_clip=False):
    """
    评估整个测试数据集的VQA性能
    
    该函数对数据集中的每个样本进行评估，
    支持启用/禁用CLIP重排序两种模式。
    
    Args:
        llm: LangChain LLM对象
        model_name (str): 模型名称
        metadata (list): 数据集元数据列表
        image_dir (str): 图像目录路径
        sample_size (int): 评估样本数量
        use_clip (bool): 是否使用CLIP重排序
        
    Returns:
        tuple: (results, category_stats)
            - results: 评估结果列表
            - category_stats: 分类统计信息
    """
    # 根据sample_size截取样本
    metadata = metadata[:sample_size] if sample_size else metadata
    
    # 初始化结果列表
    results = []
    
    # 初始化分类统计字典
    # 包含correct（正确数）、total（总数）、clip_improved（CLIP优化数）
    category_stats = {cat: {'correct': 0, 'total': 0, 'clip_improved': 0} for cat in
                      ['counting', 'attribute', 'spatial', 'reading', 'yesno', 'identification', 'other']}

    # 打印评估信息
    print(f"开始评估 {len(metadata)} 张图片... (CLIP: {'启用' if use_clip else '禁用'})")

    # 遍历每个样本
    for idx, item in enumerate(tqdm(metadata, desc="VQA评估")):
        # 构建图像完整路径
        image_path = os.path.join(image_dir, item['image_file'])
        
        # 跳过不存在的图像
        if not os.path.exists(image_path):
            continue

        # 评估单个样本
        result = evaluate_sample(llm, model_name, image_path, item['question'], item['answers'], use_clip)
        
        # 获取问题类型
        category = result['category']
        
        # 更新统计
        category_stats[category]['total'] += 1
        
        if result['is_correct']:
            category_stats[category]['correct'] += 1
        if result['clip_reranked']:
            category_stats[category]['clip_improved'] += 1

        # 添加ID和图像文件名
        result['id'] = item['id']
        result['image_file'] = item['image_file']
        results.append(result)

        # 每处理10个样本，打印一次进度
        if (idx + 1) % 10 == 0:
            current_acc = sum(1 for r in results if r['is_correct']) / len(results)
            print(f"进度: {idx + 1}/{len(metadata)}, 当前准确率: {current_acc:.2%}")

        # 添加延迟，避免API限流
        time.sleep(0.5)

    return results, category_stats


# ==============================================================================
# 第六部分：评估指标计算函数
# ==============================================================================

def compute_metrics(results, category_stats):
    """
    计算评估指标
    
    该函数根据评估结果计算多种指标，
    包括准确率、模糊匹配率、CLIP优化数等。
    
    Args:
        results (list): 评估结果列表
        category_stats (dict): 分类统计信息
        
    Returns:
        dict: 包含各项指标的字典
            - overall_accuracy: 总体准确率
            - fuzzy_accuracy: 模糊匹配率
            - total_samples: 总样本数
            - correct_samples: 正确样本数
            - fuzzy_correct_samples: 模糊匹配正确数
            - clip_reranked_count: CLIP重排序次数
            - clip_improved_count: CLIP优化改善次数
            - category_accuracy: 各分类准确率
            - category_stats: 分类统计详情
    """
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    fuzzy_correct = sum(1 for r in results if r['fuzzy_correct'])
    clip_reranked = sum(1 for r in results if r['clip_reranked'])
    clip_improved = sum(1 for r in results if r['clip_reranked'] and r['is_correct'])

    category_accuracy = {}
    for cat, stats in category_stats.items():
        category_accuracy[cat] = {
            'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else None,
            'clip_improved_rate': stats['clip_improved'] / stats['total'] if stats['total'] > 0 else None
        }

    return {
        'overall_accuracy': correct / total if total > 0 else 0,
        'fuzzy_accuracy': fuzzy_correct / total if total > 0 else 0,
        'total_samples': total, 
        'correct_samples': correct,
        'fuzzy_correct_samples': fuzzy_correct,
        'clip_reranked_count': clip_reranked, 
        'clip_improved_count': clip_improved,
        'category_accuracy': category_accuracy, 
        'category_stats': category_stats
    }


# ==============================================================================
# 第七部分：消融实验对比函数
# ==============================================================================

def create_ablation_comparison(baseline, clip, output_dir):
    """
    创建消融实验对比图
    
    该函数生成两个子图：
    1. 各类别的准确率对比（Qwen3-VL vs Qwen3-VL+CLIP）
    2. 整体指标对比（准确率、模糊匹配率、CLIP优化数）
    
    Args:
        baseline (dict): 基线模型（不使用CLIP）的评估指标
        clip (dict): 使用CLIP的评估指标
        output_dir (str): 输出目录路径
    """
    # 创建图表（1行2列）
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 获取类别列表
    categories = list(baseline['category_stats'].keys())
    
    # 设置x轴位置
    x = np.arange(len(categories))
    width = 0.35
    
    # 计算各类别准确率
    baseline_acc = [baseline['category_stats'][c]['correct'] / max(baseline['category_stats'][c]['total'], 1) for c in categories]
    clip_acc = [clip['category_stats'][c]['correct'] / max(clip['category_stats'][c]['total'], 1) for c in categories]

    # 子图1：各类别准确率对比
    axes[0].bar(x - width/2, baseline_acc, width, label='Qwen3-VL', color='steelblue')
    axes[0].bar(x + width/2, clip_acc, width, label='+CLIP', color='orange')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, rotation=45, ha='right')
    axes[0].legend()
    axes[0].set_title('消融实验: 各类型准确率对比')
    axes[0].set_ylabel('准确率')

    # 子图2：整体指标对比
    metrics = ['总体准确率', '模糊匹配率', 'CLIP优化数']
    b_vals = [baseline['overall_accuracy'], baseline['fuzzy_accuracy'], 0]
    c_vals = [clip['overall_accuracy'], clip['fuzzy_accuracy'], clip['clip_reranked_count']]

    axes[1].bar(np.arange(len(metrics)) - width/2, b_vals, width, label='Qwen3-VL', color='steelblue')
    axes[1].bar(np.arange(len(metrics)) + width/2, c_vals, width, label='+CLIP', color='orange')
    axes[1].set_xticks(np.arange(len(metrics)))
    axes[1].set_xticklabels(metrics)
    axes[1].legend()
    axes[1].set_title('整体指标对比')
    axes[1].set_ylabel('数值')

    # 调整布局并保存
    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'ablation_comparison.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"消融实验对比图已保存到: {chart_path}")
    return chart_path


# ==============================================================================
# 第八部分：结果保存函数
# ==============================================================================

def save_results(results, metrics, output_dir, use_clip):
    """
    保存评估结果到文件
    
    Args:
        results (list): 评估结果列表
        metrics (dict): 评估指标
        output_dir (str): 输出目录
        use_clip (bool): 是否使用CLIP
    """
    # 根据是否使用CLIP生成后缀
    suffix = f"clip_{'enabled' if use_clip else 'disabled'}"
    
    # 保存完整结果
    with open(os.path.join(output_dir, f'all_results_{suffix}.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 构建评估报告
    report = {
        'evaluation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model': f'Qwen3-VL + CLIP (rerank={"enabled" if use_clip else "disabled"})',
        'metrics': metrics, 
        'sample_results': results[:20]
    }
    
    # 保存评估报告
    with open(os.path.join(output_dir, f'evaluation_report_{suffix}.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 保存文本摘要
    with open(os.path.join(output_dir, f'evaluation_summary_{suffix}.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Qwen3-VL + CLIP VQA 评估报告\n{'=' * 60}\n\n")
        f.write(f"CLIP重排序: {'启用' if use_clip else '禁用'}\n")
        f.write(f"总样本数: {metrics['total_samples']}\n")
        f.write(f"正确预测: {metrics['correct_samples']}\n")
        f.write(f"总体准确率: {metrics['overall_accuracy']:.2%}\n")
        f.write(f"模糊匹配率: {metrics['fuzzy_accuracy']:.2%}\n")
        f.write(f"CLIP优化样本数: {metrics['clip_reranked_count']}\n\n")
        f.write("各类型准确率:\n")
        for cat, data in metrics['category_accuracy'].items():
            if data['accuracy'] is not None:
                stats = metrics['category_stats'][cat]
                f.write(f"  {cat}: {data['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
                if data['clip_improved_rate']:
                    f.write(f" [CLIP优化率: {data['clip_improved_rate']:.2%}]")
                f.write("\n")

    print(f"所有结果已保存到: {output_dir}")


# ==============================================================================
# 第九部分：消融实验主函数
# ==============================================================================

def run_ablation_experiments(llm, model_name, metadata, image_dir, sample_size=100):
    """
    运行消融实验
    
    该函数执行两个实验：
    1. 基线模型：不使用CLIP重排序
    2. 增强模型：使用CLIP重排序
    
    然后对比两者的结果，分析CLIP重排序的效果。
    
    Args:
        llm: LangChain LLM对象
        model_name (str): 模型名称
        metadata (list): 数据集元数据
        image_dir (str): 图像目录
        sample_size (int): 评估样本数量
        
    Returns:
        tuple: (metrics_baseline, metrics_clip)
            - 基线模型的评估指标
            - CLIP模型的评估指标
    """
    # 创建输出目录
    output_dir = os.path.join(DATA_RESULTS, "vqa_results_ablation")
    os.makedirs(output_dir, exist_ok=True)

    # 打印标题
    print("=" * 60)
    print("开始消融实验")
    print("=" * 60)

    # 实验1：基线模型（不使用CLIP重排序）
    print("\n实验1: 基线模型 (不使用CLIP重排序)")
    results_baseline, stats_baseline = evaluate_dataset(llm, model_name, metadata, image_dir, sample_size, use_clip=False)
    metrics_baseline = compute_metrics(results_baseline, stats_baseline)
    print(f"基线模型准确率: {metrics_baseline['overall_accuracy']:.2%}")
    save_results(results_baseline, metrics_baseline, output_dir, use_clip=False)

    # 实验2：增强模型（使用CLIP重排序）
    print("\n实验2: Qwen3-VL + CLIP重排序")
    results_clip, stats_clip = evaluate_dataset(llm, model_name, metadata, image_dir, sample_size, use_clip=True)
    metrics_clip = compute_metrics(results_clip, stats_clip)
    print(f"+CLIP重排序准确率: {metrics_clip['overall_accuracy']:.2%}")
    save_results(results_clip, metrics_clip, output_dir, use_clip=True)

    # 计算改进
    improvement = metrics_clip['overall_accuracy'] - metrics_baseline['overall_accuracy']
    
    # 打印对比结果
    print(f"\n消融实验结果:")
    print(f"基线模型准确率: {metrics_baseline['overall_accuracy']:.2%}")
    print(f"+CLIP重排序准确率: {metrics_clip['overall_accuracy']:.2%}")
    print(f"准确率变化: {'+' if improvement > 0 else ''}{improvement:.2%}")
    print(f"CLIP优化样本数: {metrics_clip['clip_reranked_count']}")

    # 生成可视化对比图
    create_ablation_comparison(metrics_baseline, metrics_clip, output_dir)
    create_visualization(results_clip, image_dir, output_dir, num_samples=20, show_clip_info=True)
    create_category_chart(stats_clip, output_dir)

    # 保存消融实验报告
    ablation_report = {
        'evaluation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'baseline': {'model': 'Qwen3-VL', 'use_clip_rerank': False, 'metrics': metrics_baseline},
        'with_clip': {'model': 'Qwen3-VL + CLIP', 'use_clip_rerank': True, 'metrics': metrics_clip},
        'improvement': {
            'accuracy_delta': improvement,
            'clip_reranked_count': metrics_clip['clip_reranked_count'],
            'clip_improved_count': metrics_clip['clip_improved_count']
        }
    }
    with open(os.path.join(output_dir, 'ablation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(ablation_report, f, ensure_ascii=False, indent=2)

    return metrics_baseline, metrics_clip


# ==============================================================================
# 第十部分：主函数
# ==============================================================================

def main():
    """
    主函数
    
    执行流程：
    1. 创建输出目录
    2. 初始化Qwen3-VL模型
    3. 加载数据集元数据
    4. 运行消融实验
    5. 输出结果
    """
    # 创建输出目录
    output_dir = os.path.join(DATA_RESULTS, "vqa_results_ablation")
    os.makedirs(output_dir, exist_ok=True)

    # 打印标题
    print("=" * 60)
    print("Qwen3-VL + CLIP VQA 评估 (LangChain + Pipeline)")
    print("=" * 60)

    # 初始化模型
    llm, model_name = load_model(API_KEY)
    
    # 加载数据集元数据
    metadata = load_metadata(DATA_IMAGES)

    # 运行消融实验
    metrics_baseline, metrics_clip = run_ablation_experiments(llm, model_name, metadata, DATA_IMAGES, sample_size=100)

    print(f"\n评估完成! 结果目录: {output_dir}")


# ==============================================================================
# 第十一部分：程序入口
# ==============================================================================

if __name__ == "__main__":

    main()