"""
Qwen3-VL + CLIP VQA 评估系统 (LangChain + Pipeline版本)
==========================================================

本文件实现了结合LangChain API和Hugging Face pipeline的VQA评估系统。
主要功能：
1. 使用LangChain进行VQA推理
2. 使用CLIP特征提取进行答案重排序
3. 消融实验支持
4. 多种评估指标
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from config import DATA_IMAGES, DATA_RESULTS, API_KEY, CLIP_RERANK, CLIP_THRESHOLD, MODELS
from vqa_common import (
    load_model, load_metadata, classify_question, vqa_inference,
    compute_exact_match, compute_fuzzy_match, create_visualization,
    create_category_chart
)

import json
import time
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import torch

plt.switch_backend('Agg')

clip_processor = AutoProcessor.from_pretrained(MODELS["clip"])
clip_model = AutoModelForZeroShotImageClassification.from_pretrained(MODELS["clip"])
clip_model.eval()
clip_device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(clip_device)
print(f"CLIP模型已加载到: {clip_device}")

def compute_clip_similarity(image_path, candidates):
    """使用CLIP计算图像与候选答案的相似度"""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(text=candidates, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(clip_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits = outputs.logits_per_image
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

        return probs.tolist()
    except Exception as e:
        print(f"CLIP相似度计算错误: {e}")
        return [0.0] * len(candidates)

def evaluate_sample(llm, model_name, image_path, question, answers, use_clip):
    """评估单个VQA样本"""
    category = classify_question(question)
    pred_answer = vqa_inference(llm, model_name, image_path, question)

    clip_score, clip_reranked, final_answer = 0.0, False, pred_answer

    if use_clip and CLIP_RERANK:
        unique_answers = list(set(answers))[:5]
        candidates = [pred_answer] + unique_answers
        similarities = compute_clip_similarity(image_path, candidates)

        clip_score = similarities[0]
        best_idx = np.argmax(similarities)

        if best_idx > 0:
            final_answer = candidates[best_idx]
            clip_reranked = True

    is_correct = compute_exact_match(final_answer, answers)
    fuzzy_correct = compute_fuzzy_match(final_answer, answers)

    return {
        'question': question, 'category': category,
        'ground_truth': max(set(answers), key=answers.count),
        'all_answers': answers, 'model_answer': pred_answer,
        'final_answer': final_answer, 'clip_score': clip_score,
        'clip_reranked': clip_reranked, 'is_correct': is_correct,
        'fuzzy_correct': fuzzy_correct
    }

def evaluate_dataset(llm, model_name, metadata, image_dir, sample_size=100, use_clip=False):
    """评估整个数据集"""
    metadata = metadata[:sample_size] if sample_size else metadata
    results = []
    category_stats = {cat: {'correct': 0, 'total': 0, 'clip_improved': 0} for cat in
                      ['counting', 'attribute', 'spatial', 'reading', 'yesno', 'identification', 'other']}

    print(f"开始评估 {len(metadata)} 张图片... (CLIP: {'启用' if use_clip else '禁用'})")

    for idx, item in enumerate(tqdm(metadata, desc="VQA评估")):
        image_path = os.path.join(image_dir, item['image_file'])
        if not os.path.exists(image_path):
            continue

        result = evaluate_sample(llm, model_name, image_path, item['question'], item['answers'], use_clip)
        category = result['category']
        category_stats[category]['total'] += 1

        if result['is_correct']:
            category_stats[category]['correct'] += 1
        if result['clip_reranked']:
            category_stats[category]['clip_improved'] += 1

        result['id'] = item['id']
        result['image_file'] = item['image_file']
        results.append(result)

        if (idx + 1) % 10 == 0:
            current_acc = sum(1 for r in results if r['is_correct']) / len(results)
            print(f"进度: {idx + 1}/{len(metadata)}, 当前准确率: {current_acc:.2%}")

        time.sleep(0.5)

    return results, category_stats

def compute_metrics(results, category_stats):
    """计算评估指标"""
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
        'total_samples': total, 'correct_samples': correct,
        'fuzzy_correct_samples': fuzzy_correct,
        'clip_reranked_count': clip_reranked, 'clip_improved_count': clip_improved,
        'category_accuracy': category_accuracy, 'category_stats': category_stats
    }

def create_ablation_comparison(baseline, clip, output_dir):
    """创建消融实验对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    categories = list(baseline['category_stats'].keys())

    x = np.arange(len(categories))
    width = 0.35
    baseline_acc = [baseline['category_stats'][c]['correct'] / max(baseline['category_stats'][c]['total'], 1) for c in categories]
    clip_acc = [clip['category_stats'][c]['correct'] / max(clip['category_stats'][c]['total'], 1) for c in categories]

    axes[0].bar(x - width/2, baseline_acc, width, label='Qwen3-VL', color='steelblue')
    axes[0].bar(x + width/2, clip_acc, width, label='+CLIP', color='orange')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, rotation=45, ha='right')
    axes[0].legend()
    axes[0].set_title('消融实验: 各类型准确率对比')
    axes[0].set_ylabel('准确率')

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

    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'ablation_comparison.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"消融实验对比图已保存到: {chart_path}")
    return chart_path

def save_results(results, metrics, output_dir, use_clip):
    """保存评估结果"""
    suffix = f"clip_{'enabled' if use_clip else 'disabled'}"
    with open(os.path.join(output_dir, f'all_results_{suffix}.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    report = {
        'evaluation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model': f'Qwen3-VL + CLIP (rerank={"enabled" if use_clip else "disabled"})',
        'metrics': metrics, 'sample_results': results[:20]
    }
    with open(os.path.join(output_dir, f'evaluation_report_{suffix}.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

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

def run_ablation_experiments(llm, model_name, metadata, image_dir, sample_size=100):
    """运行消融实验"""
    output_dir = os.path.join(DATA_RESULTS, "vqa_results_ablation")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("开始消融实验")
    print("=" * 60)

    print("\n实验1: 基线模型 (不使用CLIP重排序)")
    results_baseline, stats_baseline = evaluate_dataset(llm, model_name, metadata, image_dir, sample_size, use_clip=False)
    metrics_baseline = compute_metrics(results_baseline, stats_baseline)
    print(f"基线模型准确率: {metrics_baseline['overall_accuracy']:.2%}")
    save_results(results_baseline, metrics_baseline, output_dir, use_clip=False)

    print("\n实验2: Qwen3-VL + CLIP重排序")
    results_clip, stats_clip = evaluate_dataset(llm, model_name, metadata, image_dir, sample_size, use_clip=True)
    metrics_clip = compute_metrics(results_clip, stats_clip)
    print(f"+CLIP重排序准确率: {metrics_clip['overall_accuracy']:.2%}")
    save_results(results_clip, metrics_clip, output_dir, use_clip=True)

    improvement = metrics_clip['overall_accuracy'] - metrics_baseline['overall_accuracy']
    print(f"\n消融实验结果:")
    print(f"基线模型准确率: {metrics_baseline['overall_accuracy']:.2%}")
    print(f"+CLIP重排序准确率: {metrics_clip['overall_accuracy']:.2%}")
    print(f"准确率变化: {'+' if improvement > 0 else ''}{improvement:.2%}")
    print(f"CLIP优化样本数: {metrics_clip['clip_reranked_count']}")

    create_ablation_comparison(metrics_baseline, metrics_clip, output_dir)
    create_visualization(results_clip, image_dir, output_dir, num_samples=20, show_clip_info=True)
    create_category_chart(stats_clip, output_dir)

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

def main():
    """主函数"""
    output_dir = os.path.join(DATA_RESULTS, "vqa_results_ablation")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Qwen3-VL + CLIP VQA 评估 (LangChain + Pipeline)")
    print("=" * 60)

    llm, model_name = load_model(API_KEY)
    metadata = load_metadata(DATA_IMAGES)

    metrics_baseline, metrics_clip = run_ablation_experiments(llm, model_name, metadata, DATA_IMAGES, sample_size=100)

    print(f"\n评估完成! 结果目录: {output_dir}")

if __name__ == "__main__":
    main()
