"""
================================================================================
Qwen3-VL VQA 评估系统脚本
================================================================================

功能概览：
1. 数据集评估
2. 指标计算
3. 结果保存
4. 可视化

================================================================================
"""

import os
import json
import time
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from config import DATA_IMAGES, DATA_RESULTS, API_KEY

from vqa_common import (
    load_model,
    load_metadata,
    compute_accuracy,
    vqa_inference_with_temperature,
    compute_metrics_base,
    create_visualization,
    create_category_chart
)

from tqdm import tqdm

def evaluate_dataset(llm, model_name, metadata, image_dir, sample_size=100):
    """评估整个测试数据集的VQA性能"""
    metadata = metadata[:sample_size] if sample_size else metadata
    
    results = []
    
    print(f"开始评估 {len(metadata)} 张图片...")

    for idx, item in enumerate(tqdm(metadata, desc="VQA评估")):
        image_path = os.path.join(image_dir, item['image_file'])
        
        if not os.path.exists(image_path):
            continue

        pred_answer = vqa_inference_with_temperature(llm, model_name, image_path, item['question'], temperature=0.1)
        
        is_correct = compute_accuracy(pred_answer, item['answers'])

        most_common_answer = max(set(item['answers']), key=item['answers'].count)

        results.append({
            'id': item['id'], 
            'image_file': item['image_file'],
            'question': item['question'], 
            'ground_truth': most_common_answer, 
            'all_answers': item['answers'],
            'model_answer': pred_answer, 
            'is_correct': is_correct
        })

        if (idx + 1) % 10 == 0:
            current_acc = sum(1 for r in results if r['is_correct']) / len(results)
            print(f"进度: {idx + 1}/{len(metadata)}, 当前准确率: {current_acc:.2%}")

        time.sleep(0.5)

    return results


def save_results(results, metrics, output_dir):
    """保存评估结果到文件"""
    with open(os.path.join(output_dir, 'all_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    report = {
        'evaluation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model': 'qwen3-vl-8b-instruct (LangChain)',
        'metrics': metrics, 
        'sample_results': results[:20]
    }
    
    with open(os.path.join(output_dir, 'evaluation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, 'evaluation_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("Qwen3-VL VQA 评估报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"总样本数: {metrics['total_samples']}\n")
        f.write(f"正确预测: {metrics['correct_samples']}\n")
        f.write(f"总体准确率: {metrics['overall_accuracy']:.2%}\n\n")

    print(f"所有结果已保存到: {output_dir}")


def main():
    """主评估函数"""
    output_dir = os.path.join(DATA_RESULTS, "vqa_results_api")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("Qwen3-VL VQA 评估")
    print("=" * 50)

    llm, model_name = load_model(API_KEY)
    metadata = load_metadata(DATA_IMAGES)

    results = evaluate_dataset(llm, model_name, metadata, DATA_IMAGES, sample_size=100)
    metrics = compute_metrics_base(results)

    print(f"\n评估结果:")
    print(f"总体准确率: {metrics['overall_accuracy']:.2%}")

    create_visualization(results, DATA_IMAGES, output_dir)
    create_category_chart(results, output_dir)
    save_results(results, metrics, output_dir)

    print(f"\n评估完成! 结果目录: {output_dir}")


if __name__ == "__main__":
    main()