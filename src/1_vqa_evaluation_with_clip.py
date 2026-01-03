"""
================================================================================
Qwen3-VL + CLIP VQA 评估系统脚本
================================================================================
功能概览：
1. CLIP模型集成
2. 答案重排序
3. 消融实验
4. 多指标评估
5. 可视化对比
================================================================================
"""

import os
import json
import time
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from config import DATA_IMAGES, DATA_RESULTS, API_KEY, CLIP_RERANK, CLIP_THRESHOLD, MODELS

from vqa_common import (
    load_model,
    load_metadata,
    vqa_inference_with_temperature,
    generate_multiple_candidates,
    compute_match,
    create_visualization,
    create_category_chart
)

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn.functional as F

plt.switch_backend('Agg')

clip_processor = AutoProcessor.from_pretrained(MODELS["clip"])
clip_model = AutoModel.from_pretrained(MODELS["clip"])
clip_model.eval()
clip_device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(clip_device)
print(f"CLIP模型已加载到: {clip_device}")


def compute_clip_similarity(image_path, candidates):
    """使用CLIP计算图像与候选答案的相似度"""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, text=candidates, return_tensors="pt", padding=True)
        inputs = {k: v.to(clip_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = clip_model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            
            image_embeds = F.normalize(image_embeds, p=2, dim=1)
            text_embeds = F.normalize(text_embeds, p=2, dim=1)
            
            logits_per_image = image_embeds @ text_embeds.T
            probs = logits_per_image.softmax(dim=1).cpu().numpy().flatten()

        return probs.tolist()
        
    except Exception as e:
        print(f"CLIP相似度计算错误: {e}")
        import traceback
        traceback.print_exc()
        return [0.0] * len(candidates)


def evaluate_sample(llm, model_name, image_path, question, answers, use_clip):
    """评估单个VQA样本（支持CLIP重排序）"""
    pred_answer = vqa_inference_with_temperature(llm, model_name, image_path, question, temperature=0.1)

    clip_score, clip_reranked, final_answer = 0.0, False, pred_answer
    clip_debug_info = None

    if use_clip and CLIP_RERANK:
        candidate_answers = generate_multiple_candidates(
            llm, model_name, image_path, question, num_candidates=10
        )
        
        candidates = [pred_answer] + [ans for ans in candidate_answers if ans != pred_answer]
        candidates = candidates[:10]
        
        if len(candidates) > 1:
            similarities = compute_clip_similarity(image_path, candidates)
            clip_score = similarities[0]
            best_idx = np.argmax(similarities)
            
            sorted_indices = np.argsort(similarities)[::-1]
            
            clip_debug_info = {
                'candidates': candidates,
                'similarities': similarities,
                'best_idx': best_idx,
                'best_candidate': candidates[best_idx] if best_idx < len(candidates) else None
            }

            top1_idx = sorted_indices[0]
            top2_idx = sorted_indices[1] if len(sorted_indices) > 1 else sorted_indices[0]
            
            confidence_margin = similarities[top1_idx] - similarities[top2_idx]
            original_similarity = similarities[candidates.index(pred_answer)] if pred_answer in candidates else 0
            
            if top1_idx != 0 and confidence_margin > 0.02:
                final_answer = candidates[top1_idx]
                clip_reranked = True

    is_correct = compute_match(final_answer, answers)

    return {
        'question': question, 
        'ground_truth': max(set(answers), key=answers.count),
        'all_answers': answers, 
        'model_answer': pred_answer,
        'final_answer': final_answer, 
        'clip_score': clip_score,
        'clip_reranked': clip_reranked, 
        'clip_debug': clip_debug_info,
        'is_correct': is_correct
    }


def evaluate_dataset(llm, model_name, metadata, image_dir, sample_size=100, use_clip=False):
    """评估整个测试数据集的VQA性能"""
    metadata = metadata[:sample_size] if sample_size else metadata
    
    results = []
    correct_count = 0
    clip_improved_count = 0

    print(f"开始评估 {len(metadata)} 张图片... (CLIP: {'启用' if use_clip else '禁用'})")

    for idx, item in enumerate(tqdm(metadata, desc="VQA评估")):
        image_path = os.path.join(image_dir, item['image_file'])
        
        if not os.path.exists(image_path):
            continue

        result = evaluate_sample(llm, model_name, image_path, item['question'], item['answers'], use_clip)
        
        if result['is_correct']:
            correct_count += 1
        if result['clip_reranked']:
            clip_improved_count += 1
            debug = result.get('clip_debug')
            if debug:
                print(f"\n[CLIP重排序 #{idx}] 原始答案: {result['model_answer']}")
                print(f"  候选答案数量: {len(debug['candidates'])}")
                print(f"  相似度得分: {debug['similarities']}")
                print(f"  选择的答案: {debug['best_candidate']}")
                print(f"  正确答案: {result['ground_truth']}")

        result['id'] = item['id']
        result['image_file'] = item['image_file']
        results.append(result)

        if (idx + 1) % 10 == 0:
            current_acc = sum(1 for r in results if r['is_correct']) / len(results)
            print(f"进度: {idx + 1}/{len(metadata)}, 当前准确率: {current_acc:.2%}")

        time.sleep(0.5)

    stats = {
        'correct': correct_count,
        'clip_improved': clip_improved_count
    }
    return results, stats


def compute_metrics(results, stats):
    """计算评估指标"""
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    clip_reranked = sum(1 for r in results if r['clip_reranked'])
    clip_improved = stats.get('clip_improved', 0)

    return {
        'overall_accuracy': correct / total if total > 0 else 0,
        'total_samples': total, 
        'correct_samples': correct,
        'clip_reranked_count': clip_reranked, 
        'clip_improved_count': clip_improved
    }


def create_ablation_comparison(baseline, clip, output_dir):
    """创建消融实验对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    metrics = ['总体准确率']
    b_vals = [baseline['overall_accuracy']]
    c_vals = [clip['overall_accuracy']]

    axes[0].bar(np.arange(len(metrics)) - 0.175, b_vals, 0.35, label='Qwen3-VL', color='steelblue')
    axes[0].bar(np.arange(len(metrics)) + 0.175, c_vals, 0.35, label='+CLIP', color='orange')
    axes[0].set_xticks(np.arange(len(metrics)))
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].set_title('准确率对比')
    axes[0].set_ylabel('准确率')
    axes[0].set_ylim(0, 1)

    clip_metrics = ['CLIP重排序数', 'CLIP改善数']
    clip_vals = [clip['clip_reranked_count'], clip['clip_improved_count']]

    axes[1].bar(clip_metrics, clip_vals, color='orange')
    axes[1].set_title('CLIP优化统计')
    axes[1].set_ylabel('数量')
    for i, v in enumerate(clip_vals):
        axes[1].text(i, v + 1, str(v), ha='center')

    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'ablation_comparison.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"消融实验对比图已保存到: {chart_path}")
    return chart_path


def convert_to_native_types(obj):
    """将numpy类型转换为Python原生类型"""
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_results(results, metrics, output_dir, use_clip):
    """保存评估结果到文件"""
    results = convert_to_native_types(results)
    
    suffix = f"clip_{'enabled' if use_clip else 'disabled'}"
    
    with open(os.path.join(output_dir, f'all_results_{suffix}.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    report = {
        'evaluation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model': f'Qwen3-VL + CLIP (rerank={"enabled" if use_clip else "disabled"})',
        'metrics': metrics, 
        'sample_results': results[:20]
    }
    
    with open(os.path.join(output_dir, f'evaluation_report_{suffix}.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, f'evaluation_summary_{suffix}.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Qwen3-VL + CLIP VQA 评估报告\n{'=' * 60}\n\n")
        f.write(f"CLIP重排序: {'启用' if use_clip else '禁用'}\n")
        f.write(f"总样本数: {metrics['total_samples']}\n")
        f.write(f"正确预测: {metrics['correct_samples']}\n")
        f.write(f"总体准确率: {metrics['overall_accuracy']:.2%}\n")
        f.write(f"CLIP优化样本数: {metrics['clip_reranked_count']}\n")

    print(f"所有结果已保存到: {output_dir}")


def run_ablation_experiments(llm, model_name, metadata, image_dir, sample_size=100):
    """运行消融实验（对比有无CLIP重排序）"""
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
    create_category_chart(results_clip, output_dir)

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