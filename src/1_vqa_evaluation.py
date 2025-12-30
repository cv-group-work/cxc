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

# 导入所需的标准库
import json           # 用于处理JSON数据
import os            # 用于文件和目录操作
import time          # 用于控制时间间隔
from datetime import datetime  # 用于处理日期和时间

# 导入进度条库，用于显示评估进度
from tqdm import tqdm

# 从配置文件中导入必要的配置参数
from config import DATA_IMAGES, DATA_RESULTS, API_KEY

# 导入公共功能模块，包含VQA评估所需的核心功能
from vqa_common import (
    load_model,           # 初始化Qwen3-VL API客户端
    load_metadata,        # 加载数据集元数据
    compute_accuracy,     # 计算预测答案的准确率
    classify_question,    # 问题类型自动分类
    vqa_inference,        # 视觉问答推理函数
    compute_metrics_base,  # 计算基础评估指标
    create_visualization,  # 创建VQA评估结果的可视化展示
    create_category_chart  # 创建问题类型统计图表
)




def evaluate_dataset(client, model_name, metadata, image_dir, sample_size=100):
    """
    评估整个数据集
    
    对数据集中的样本进行批量评估，
    包含问题分类、推理、结果统计等功能。
    
    Args:
        client: API客户端
        model_name (str): 模型名称
        metadata (list): 数据集元数据，包含图像、问题和答案信息
        image_dir (str): 图像目录路径
        sample_size (int): 评估样本数量，默认100个样本
    
    Returns:
        tuple: (results, category_stats) - 评估结果和分类统计
    """
    # 根据sample_size限制评估样本数量，None或0表示评估所有样本
    metadata = metadata[:sample_size] if sample_size else metadata
    # 初始化评估结果列表
    results = []
    
    # 初始化各问题类型的统计字典
    # 键为问题类型，值为包含正确数量和总数量的字典
    category_stats = {cat: {'correct': 0, 'total': 0} for cat in
                      ['counting', 'attribute', 'spatial', 'reading', 'yesno', 'identification', 'other']}
    
    # 打印评估开始信息
    print(f"开始评估 {len(metadata)} 张图片...")
    
    # 使用tqdm显示评估进度条
    for idx, item in enumerate(tqdm(metadata, desc="VQA评估")):
        # 构建完整的图像路径
        image_path = os.path.join(image_dir, item['image_file'])
        # 检查图像文件是否存在，不存在则跳过
        if not os.path.exists(image_path):
            continue  # 跳过不存在的图像
        
        # 对问题进行分类
        category = classify_question(item['question'])
        # 更新当前问题类型的总样本数
        category_stats[category]['total'] += 1
        
        # 调用VQA推理函数，获取模型预测的答案
        pred_answer = vqa_inference(client, model_name, image_path, item['question'])
        
        # 计算预测答案的准确率
        is_correct = compute_accuracy(pred_answer, item['answers'])
        
        # 如果预测正确，更新当前问题类型的正确样本数
        if is_correct:
            category_stats[category]['correct'] += 1
        
        # 找出最常见的标注答案，作为参考标准
        most_common_answer = max(set(item['answers']), key=item['answers'].count)
        
        # 保存当前样本的详细评估结果
        results.append({
            'id': item['id'],  # 样本ID
            'image_file': item['image_file'],  # 图像文件名
            'question': item['question'],  # 问题文本
            'category': category,  # 问题类型
            'ground_truth': most_common_answer,  # 最常见的参考答案
            'all_answers': item['answers'],  # 所有参考答案列表
            'model_answer': pred_answer,  # 模型预测的答案
            'is_correct': is_correct  # 预测是否正确
        })
        
        # 每评估10个样本，打印一次当前进度和准确率
        if (idx + 1) % 10 == 0:
            # 计算当前准确率
            current_acc = sum(1 for r in results if r['is_correct']) / len(results)
            # 打印进度信息
            print(f"进度: {idx + 1}/{len(metadata)}, 当前准确率: {current_acc:.2%}")
        
        # 等待0.5秒，避免API频率限制
        time.sleep(0.5)
    
    # 返回评估结果和分类统计
    return results, category_stats




def compute_metrics(results, category_stats):
    """
    计算评估指标
    
    汇总评估结果，计算总体和各分类的准确率等指标。
    该函数是公共模块compute_metrics_base的封装函数。
    
    Args:
        results (list): 评估结果列表，包含每个样本的评估详情
        category_stats (dict): 分类统计字典，包含各问题类型的正确数量和总数量
    
    Returns:
        dict: 包含各种评估指标的字典
    """
    # 调用公共模块中的compute_metrics_base函数计算评估指标
    return compute_metrics_base(results, category_stats)


def save_results(results, metrics, output_dir):
    """
    保存评估结果到多种格式
    
    保存详细的评估结果、报告摘要和统计信息到文件。
    支持JSON格式的完整结果和报告，以及文本格式的评估摘要。
    
    Args:
        results (list): 完整评估结果列表，包含每个样本的详细评估信息
        metrics (dict): 评估指标，包含总体准确率和各类型准确率
        output_dir (str): 输出目录路径
    """
    # 保存完整结果为JSON格式，包含所有样本的评估详情
    with open(os.path.join(output_dir, 'all_results.json'), 'w', encoding='utf-8') as f:
        # 保存时确保中文正确显示，缩进为2个空格
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 创建评估报告字典，包含评估时间、模型信息、指标和示例结果
    report = {
        'evaluation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 评估时间
        'model': 'qwen3-vl-8b-instruct (API)',  # 使用的模型名称
        'metrics': metrics,  # 评估指标
        'sample_results': results[:20]  # 只保存前20个样本作为示例
    }
    # 保存评估报告为JSON格式
    with open(os.path.join(output_dir, 'evaluation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 创建文本格式的评估摘要，便于直接阅读
    with open(os.path.join(output_dir, 'evaluation_summary.txt'), 'w', encoding='utf-8') as f:
        # 写入报告标题
        f.write("Qwen3-VL VQA 评估报告 (API版本)\n")
        f.write("=" * 50 + "\n\n")
        # 写入总体评估结果
        f.write(f"总样本数: {metrics['total_samples']}\n")
        f.write(f"正确预测: {metrics['correct_samples']}\n")
        f.write(f"总体准确率: {metrics['overall_accuracy']:.2%}\n\n")
        # 写入各问题类型的准确率
        f.write("各类型准确率:\n")
        for cat, acc in metrics['category_accuracy'].items():
            if acc is not None:  # 只处理有数据的类型
                stats = metrics['category_stats'][cat]  # 获取当前类型的统计数据
                # 写入类型名称、准确率、正确数/总数
                f.write(f"  {cat}: {acc:.2%} ({stats['correct']}/{stats['total']})\n")
    
    # 打印结果保存路径
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
    
    这是整个评估系统的入口点，负责协调各个模块的执行。
    """
    # 构建输出目录路径，用于保存评估结果
    output_dir = os.path.join(DATA_RESULTS, "vqa_results_api")
    # 创建输出目录，exist_ok=True表示目录已存在时不报错
    os.makedirs(output_dir, exist_ok=True)
    
    # 打印评估开始信息，使用等号分隔线增强可读性
    print("=" * 50)
    print("Qwen3-VL VQA 评估 (API版本)")
    print("=" * 50)
    
    # 初始化Qwen3-VL API客户端
    client, model_name = load_model(API_KEY)
    
    # 加载测试数据集元数据
    metadata = load_metadata(DATA_IMAGES)
    
    # 执行数据集评估，评估100个样本
    results, category_stats = evaluate_dataset(client, model_name, metadata, DATA_IMAGES, sample_size=100)
    
    # 计算评估指标
    metrics = compute_metrics(results, category_stats)
    
    # 打印评估结果摘要
    print(f"\n评估结果:")
    print(f"总体准确率: {metrics['overall_accuracy']:.2%}")
    # 打印各问题类型的准确率
    for cat, acc in metrics['category_accuracy'].items():
        if acc is not None:
            stats = metrics['category_stats'][cat]
            print(f"  {cat}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    
    # 生成可视化图表
    # 创建VQA评估结果的可视化展示
    create_visualization(results, DATA_IMAGES, output_dir)
    # 创建问题类型统计图表
    create_category_chart(category_stats, output_dir)
    
    # 保存所有评估结果到文件
    save_results(results, metrics, output_dir)
    
    # 打印评估完成信息
    print(f"\n评估完成! 结果目录: {output_dir}")


# 当脚本直接运行时，调用main函数
if __name__ == "__main__":
    main()
