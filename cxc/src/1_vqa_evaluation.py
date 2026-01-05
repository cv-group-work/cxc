"""
================================================================================
Qwen3-VL VQA 评估系统脚本
================================================================================

本文件实现了基于Qwen3-VL API的视觉问答（VQA）评估系统。
通过对测试数据集进行批量推理和评估，计算模型的VQA性能指标。

主要功能模块：
1. 数据集评估 - 对整个测试数据集进行VQA推理和评估
2. 问题分类 - 将问题按类型分类（计数、属性、空间关系等）
3. 指标计算 - 计算精确匹配、模糊匹配等评估指标
4. 结果保存 - 保存详细评估结果和统计报告
5. 可视化 - 生成评估结果的可视化图表

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
from cxc.config import DATA_IMAGES, DATA_RESULTS, API_KEY

# 导入公共功能模块
from vqa_common import (
    load_model,              # 初始化模型客户端
    load_metadata,           # 加载数据集元数据
    compute_accuracy,        # 计算综合准确率
    classify_question,       # 问题类型分类
    vqa_inference,          # VQA推理
    compute_metrics_base,   # 计算基础评估指标
    create_visualization,   # 创建评估结果可视化
    create_category_chart   # 创建分类统计图表
)

# 导入进度条库
from tqdm import tqdm

# ==============================================================================
# 第二部分：数据集评估函数
# ==============================================================================

def evaluate_dataset(llm, model_name, metadata, image_dir, sample_size=100):
    """
    评估整个测试数据集的VQA性能
    
    该函数对数据集中的每个样本进行VQA推理，
    并根据标准答案计算评估指标。

    Args:
        llm: LangChain LLM对象
        model_name (str): 模型名称
        metadata (list): 数据集元数据列表
        image_dir (str): 图像目录路径
        sample_size (int): 评估样本数量，None表示评估全部
        
    Returns:
        tuple: 包含以下两个元素：
            - results (list): 每个样本的评估结果
            - category_stats (dict): 各问题类型的统计信息

    """
    # 根据sample_size截取样本
    metadata = metadata[:sample_size] if sample_size else metadata
    
    # 初始化结果列表
    results = []
    
    # 初始化分类统计字典
    # 包含7种预定义问题类型和1个'other'类型
    category_stats = {cat: {'correct': 0, 'total': 0} for cat in ['counting', 'attribute', 'spatial', 'reading', 'yesno', 'identification', 'other']}

    print(f"开始评估 {len(metadata)} 张图片...")

    # 遍历每个样本
    for idx, item in enumerate(tqdm(metadata, desc="VQA评估")):
        # 构建图像完整路径
        image_path = os.path.join(image_dir, item['image_file'])
        
        # 跳过不存在的图像
        if not os.path.exists(image_path):
            continue

        # 对问题进行分类
        category = classify_question(item['question'])
        category_stats[category]['total'] += 1

        # 执行VQA推理，获取预测答案
        pred_answer = vqa_inference(llm, model_name, image_path, item['question'])
        
        # 计算答案是否正确
        is_correct = compute_accuracy(pred_answer, item['answers'])

        # 如果正确，增加正确计数
        if is_correct:
            category_stats[category]['correct'] += 1

        # 获取最常见的标准答案
        most_common_answer = max(set(item['answers']), key=item['answers'].count)

        # 构建单个样本的评估结果
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

        # 每处理10个样本，打印一次进度
        if (idx + 1) % 10 == 0:
            current_acc = sum(1 for r in results if r['is_correct']) / len(results)
            print(f"进度: {idx + 1}/{len(metadata)}, 当前准确率: {current_acc:.2%}")

        # 添加延迟，避免API限流
        time.sleep(0.5)

    return results, category_stats


# ==============================================================================
# 第三部分：结果保存函数
# ==============================================================================

def save_results(results, metrics, output_dir):
    """
    保存评估结果到文件
    
    该函数将评估结果保存为多种格式：
    1. JSON格式的完整结果
    2. JSON格式的评估报告
    3. 文本格式的评估摘要
    
    Args:
        results (list): 完整评估结果列表
        metrics (dict): 评估指标字典
        output_dir (str): 输出目录路径
    """
    # 保存完整评估结果为JSON
    with open(os.path.join(output_dir, 'all_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 构建评估报告
    report = {
        'evaluation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model': 'qwen3-vl-8b-instruct (LangChain)',
        'metrics': metrics, 
        'sample_results': results[:20]  # 只保存前20个样本的详细结果
    }
    
    # 保存评估报告为JSON
    with open(os.path.join(output_dir, 'evaluation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 保存评估摘要为文本
    with open(os.path.join(output_dir, 'evaluation_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("Qwen3-VL VQA 评估报告\n")
        f.write("=" * 50 + "\n\n")
        
        # 总体统计
        f.write(f"总样本数: {metrics['total_samples']}\n")
        f.write(f"正确预测: {metrics['correct_samples']}\n")
        f.write(f"总体准确率: {metrics['overall_accuracy']:.2%}\n\n")
        
        # 各类型详细统计
        f.write("各类型准确率:\n")
        for cat, acc in metrics['category_accuracy'].items():
            if acc is not None:
                stats = metrics['category_stats'][cat]
                f.write(f"  {cat}: {acc:.2%} ({stats['correct']}/{stats['total']})\n")

    print(f"所有结果已保存到: {output_dir}")


# ==============================================================================
# 第四部分：主函数
# ==============================================================================

def main():
    """
    主评估函数
    
    执行流程：
    1. 创建输出目录
    2. 初始化Qwen3-VL模型客户端
    3. 加载数据集元数据
    4. 执行数据集评估
    5. 计算评估指标
    6. 生成可视化结果
    7. 保存评估结果
    """
    # 创建输出目录
    output_dir = os.path.join(DATA_RESULTS, "vqa_results_api")
    os.makedirs(output_dir, exist_ok=True)

    # 打印标题
    print("=" * 50)
    print("Qwen3-VL VQA 评估")
    print("=" * 50)

    # 初始化模型
    llm, model_name = load_model(API_KEY)
    
    # 加载数据集元数据
    metadata = load_metadata(DATA_IMAGES)

    # 执行评估
    results, category_stats = evaluate_dataset(llm, model_name, metadata, DATA_IMAGES, sample_size=100)
    
    # 计算评估指标
    metrics = compute_metrics_base(results, category_stats)

    # 打印评估结果
    print(f"\n评估结果:")
    print(f"总体准确率: {metrics['overall_accuracy']:.2%}")
    for cat, acc in metrics['category_accuracy'].items():
        if acc is not None:
            stats = metrics['category_stats'][cat]
            print(f"  {cat}: {acc:.2%} ({stats['correct']}/{stats['total']})")

    # 生成可视化结果
    create_visualization(results, DATA_IMAGES, output_dir)
    create_category_chart(category_stats, output_dir)
    
    # 保存评估结果
    save_results(results, metrics, output_dir)

    print(f"\n评估完成! 结果目录: {output_dir}")


# ==============================================================================
# 第五部分：程序入口
# ==============================================================================

if __name__ == "__main__":
    main()