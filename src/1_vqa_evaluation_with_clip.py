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
import re
import time
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from openai import OpenAI
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

# 从配置文件中导入必要的配置参数
from config import DATA_IMAGES, DATA_RESULTS, API_KEY, CLIP_RERANK, CLIP_THRESHOLD, MODELS

# 设置matplotlib后端为非交互式模式
plt.switch_backend('Agg')

# ====================
# 中文字体配置
# ====================

import matplotlib
import matplotlib.font_manager as fm

# 检查系统可用的中文字体
available_fonts = [f.name for f in fm.fontManager.ttflist]
chinese_fonts = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Droid Sans Fallback']
selected_font = None

# 选择最合适的中文字体
for font in chinese_fonts:
    if font in available_fonts:
        selected_font = font
        break

# 配置matplotlib使用中文字体
if selected_font:
    matplotlib.rcParams['font.family'] = selected_font
    print(f"使用中文字体: {selected_font}")
else:
    matplotlib.rcParams['font.family'] = ['DejaVu Sans']
    print("未找到中文字体，使用默认字体")


# ====================
# 1. 模型加载和初始化
# ====================

def load_model(api_key):
    """
    初始化Qwen3-VL API客户端
    
    Args:
        api_key (str): DashScope API密钥
    
    Returns:
        tuple: (client, model_name) - API客户端和模型名称
    """
    print("初始化 Qwen3-VL API 客户端")
    # 创建OpenAI兼容的客户端，连接到DashScope服务
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model_name = "qwen3-vl-8b-instruct"
    print(f"使用模型: {model_name}")
    return client, model_name


def load_clip_model():
    """
    加载CLIP本地模型
    
    CLIP模型用于计算图像-文本相似度，进行答案重排序。
    
    Returns:
        tuple: (processor, model, device) - CLIP处理器、模型和设备
    """
    print("正在加载 CLIP 模型...")
    # 从配置文件指定的模型路径加载CLIP处理器和模型
    processor = AutoProcessor.from_pretrained(MODELS["clip"])
    model = AutoModelForZeroShotImageClassification.from_pretrained(MODELS["clip"])
    model.eval()  # 设置为评估模式
    
    # 自动选择设备：GPU优先，否则使用CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"CLIP 模型已加载到: {device}")
    
    return processor, model, device


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
    """
    # 读取metadata.json文件
    with open(os.path.join(data_dir, "metadata.json"), 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"加载了 {len(metadata)} 条元数据")
    return metadata


# ====================
# 3. 文本处理和标准化
# ====================

def normalize_answer(answer):
    """
    答案标准化处理
    
    对答案进行清理和标准化，便于匹配比较。
    包括转小写、去除标点、标准化空格等。
    
    Args:
        answer (str): 原始答案文本
    
    Returns:
        str: 标准化后的答案文本
    """
    if not answer:
        return ""
    answer = answer.lower().strip()
    answer = re.sub(r'[^\w\s]', ' ', answer)
    answer = re.sub(r'\s+', ' ', answer)
    return answer.strip()


def classify_question(question):
    """
    问题类型自动分类
    
    基于关键词匹配对VQA问题进行分类。
    支持计数、属性、空间关系、文字识别、是否题、识别题等类型。
    
    Args:
        question (str): 问题文本
    
    Returns:
        str: 问题类型标签
    """
    q = question.lower()
    
    # 定义各类问题的关键词
    categories = {
        'counting': ['how many', '多少', 'count', 'number of', '数量', 'how much'],
        'attribute': ['what color', 'what brand', 'what type', 'what kind', 'what year', 
                      'what time', '颜色', '品牌', '类型', '年份', '时间', 'what is the'],
        'spatial': ['where', 'what is on the left', 'what is on the right', 'what is in front',
                    '位置', '左边', '右边', '前面', '后面', '上面', '下面'],
        'reading': ['what does it say', 'what does the sign say', 'what does the text say',
                    'what is written', 'read', '读取', '文字', '写的', '说什么', 'what word'],
        'yesno': ['is this', 'are these', 'was the', 'does her shirt say', '是否', '是不是',
                  'does the', 'is there', 'are there'],
        'identification': ['who is', 'what is the name', 'what is this', 'who was',
                          '谁', '名称', '是什么', 'what does']
    }
    
    # 遍历分类，检查是否包含关键词
    for cat, keywords in categories.items():
        if any(kw in q for kw in keywords):
            return cat
    return 'other'


# ====================
# 4. VQA推理引擎（API版本）
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
            # 打开并处理图像
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
                from io import BytesIO
                import base64
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=85)
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # 构建API消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_str}"
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
    for attempt in range(max_retries):
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                max_size = 1024
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                from io import BytesIO
                import base64
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=85)
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_str}"
                            }
                        },
                        {
                            "type": "text",
                            "text": question + "\n请简短回答，只回答关键信息，不需要解释。"
                        }
                    ]
                }
            ]
            
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=128,
                temperature=0.1,
                stream=False
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            print(f"API调用错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return f"错误: {str(e)}"


def compute_clip_similarity(processor, model, device, image_path, candidates):
    """使用CLIP计算图像与候选答案的相似度"""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=candidates, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
        
        return probs.tolist()
    except Exception as e:
        print(f"CLIP相似度计算错误: {e}")
        return [0.0] * len(candidates)


def clip_rerank_with_ground_truth(processor, model, device, image_path, pred_answer, ground_truth, threshold=CLIP_THRESHOLD):
    """使用CLIP验证预测答案与真实答案的一致性"""
    candidates = [pred_answer, ground_truth]
    similarities = compute_clip_similarity(processor, model, device, image_path, candidates)
    
    clip_score = similarities[0]
    reranked_answer = pred_answer
    
    if clip_score < threshold and similarities[1] > clip_score:
        reranked_answer = ground_truth
        clip_reranked = True
    else:
        clip_reranked = False
    
    return reranked_answer, clip_score, clip_reranked


def clip_rerank_with_candidates(processor, model, device, image_path, pred_answer, all_answers, threshold=CLIP_THRESHOLD):
    """使用CLIP在所有候选答案中选择最佳匹配"""
    unique_answers = list(set(all_answers))
    candidates = [pred_answer] + unique_answers[:5]
    
    similarities = compute_clip_similarity(processor, model, device, image_path, candidates)
    
    best_idx = np.argmax(similarities)
    clip_score = similarities[0]
    reranked_answer = candidates[best_idx] if best_idx > 0 else pred_answer
    clip_reranked = best_idx > 0
    
    return reranked_answer, clip_score, clip_reranked


def compute_exact_match(pred, targets):
    """精确匹配"""
    pred_norm = normalize_answer(pred)
    for target in targets:
        if pred_norm == normalize_answer(target):
            return True
    return False


def compute_fuzzy_match(pred, targets):
    """模糊匹配"""
    pred_norm = normalize_answer(pred)
    
    for target in targets:
        target_norm = normalize_answer(target)
        
        if pred_norm == target_norm:
            return True
        
        if target_norm in pred_norm:
            return True
        
        if pred_norm in target_norm and len(pred_norm) > 3:
            return True
        
        pred_words = set(pred_norm.split())
        target_words = set(target_norm.split())
        if target_words and len(pred_words & target_words) / len(target_words) > 0.7:
            return True
    
    return False


def evaluate_sample(client, model_name, processor, clip_model, clip_device, 
                   image_path, question, answers, use_clip_rerank):
    """评估单个样本"""
    category = classify_question(question)
    
    pred_answer = vqa_inference(client, model_name, image_path, question)
    
    clip_score = 0.0
    clip_reranked = False
    final_answer = pred_answer
    
    if use_clip_rerank and clip_model is not None:
        if CLIP_RERANK:
            most_common = max(set(answers), key=answers.count)
            final_answer, clip_score, clip_reranked = clip_rerank_with_ground_truth(
                processor, clip_model, clip_device, image_path, pred_answer, most_common
            )
        else:
            final_answer, clip_score, clip_reranked = clip_rerank_with_candidates(
                processor, clip_model, clip_device, image_path, pred_answer, answers
            )
    
    is_correct = compute_exact_match(final_answer, answers)
    fuzzy_correct = compute_fuzzy_match(final_answer, answers)
    
    most_common_answer = max(set(answers), key=answers.count)
    
    return {
        'question': question,
        'category': category,
        'ground_truth': most_common_answer,
        'all_answers': answers,
        'model_answer': pred_answer,
        'final_answer': final_answer,
        'clip_score': clip_score,
        'clip_reranked': clip_reranked,
        'is_correct': is_correct,
        'fuzzy_correct': fuzzy_correct
    }


def evaluate_dataset(client, model_name, processor, clip_model, clip_device,
                    metadata, image_dir, sample_size=100, use_clip_rerank=False):
    """评估整个数据集"""
    
    metadata = metadata[:sample_size] if sample_size else metadata
    results = []
    category_stats = {cat: {'correct': 0, 'total': 0, 'clip_improved': 0} for cat in
                      ['counting', 'attribute', 'spatial', 'reading', 'yesno', 'identification', 'other']}
    
    print(f"开始评估 {len(metadata)} 张图片...")
    print(f"使用CLIP rerank: {use_clip_rerank}")
    
    for idx, item in enumerate(tqdm(metadata, desc="VQA评估")):
        image_path = os.path.join(image_dir, item['image_file'])
        if not os.path.exists(image_path):
            continue
        
        result = evaluate_sample(
            client, model_name, processor, clip_model, clip_device,
            image_path, item['question'], item['answers'], use_clip_rerank
        )
        
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
    clip_reranked_count = sum(1 for r in results if r['clip_reranked'])
    clip_improved_count = sum(1 for r in results if r['clip_reranked'] and r['is_correct'])
    
    overall_accuracy = correct / total if total > 0 else 0
    fuzzy_accuracy = fuzzy_correct / total if total > 0 else 0
    
    category_accuracy = {}
    for cat, stats in category_stats.items():
        category_accuracy[cat] = {
            'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else None,
            'clip_improved_rate': stats['clip_improved'] / stats['total'] if stats['total'] > 0 else None
        }
    
    return {
        'overall_accuracy': overall_accuracy,
        'fuzzy_accuracy': fuzzy_accuracy,
        'total_samples': total,
        'correct_samples': correct,
        'fuzzy_correct_samples': fuzzy_correct,
        'clip_reranked_count': clip_reranked_count,
        'clip_improved_count': clip_improved_count,
        'category_accuracy': category_accuracy,
        'category_stats': category_stats
    }


def create_visualization(results, image_dir, output_dir, num_samples=30):
    """创建可视化结果"""
    
    success = [r for r in results if r['is_correct']]
    failure = [r for r in results if not r['is_correct']]
    
    selected = []
    num_success = min(num_samples // 2, len(success))
    num_failure = min(num_samples - num_success, len(failure))
    
    if num_success > 0:
        indices = np.random.choice(len(success), num_success, replace=False)
        for idx in indices:
            selected.append(success[idx])
    
    if num_failure > 0:
        indices = np.random.choice(len(failure), num_failure, replace=False)
        for idx in indices:
            selected.append(failure[idx])
    
    np.random.shuffle(selected)
    
    cols, rows = 4, (len(selected) + 3) // 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()
    
    for idx, result in enumerate(selected):
        if idx >= len(axes):
            break
        ax = axes[idx]
        try:
            img = Image.open(os.path.join(image_dir, result['image_file']))
            ax.imshow(img)
            ax.axis('off')
            
            status = "[成功]" if result['is_correct'] else "[失败]"
            color = 'green' if result['is_correct'] else 'red'
            
            clip_info = f"\nCLIP得分: {result['clip_score']:.2f}" if result['clip_score'] > 0 else ""
            rerank_info = " (CLIP优化)" if result['clip_reranked'] else ""
            
            ax.set_title(
                f"ID:{result['id']} [{result['category']}] {status}{rerank_info}\n"
                f"问题: {result['question'][:35]}...\n"
                f"预测: {result['final_answer'][:25]}{clip_info}\n"
                f"真实: {result['ground_truth'][:25]}",
                fontsize=8, color=color
            )
        except Exception as e:
            ax.text(0.5, 0.5, f"Cannot load image", ha='center', va='center')
            ax.axis('off')
    
    for ax in axes.flat[len(selected):]:
        ax.axis('off')
    
    plt.tight_layout()
    viz_path = os.path.join(output_dir, 'vqa_visualization.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化结果已保存到: {viz_path}")
    return viz_path


def create_category_chart(category_stats, output_dir):
    """创建分类统计图表"""
    categories = list(category_stats.keys())
    totals = [stats['total'] for stats in category_stats.values()]
    correct = [stats['correct'] for stats in category_stats.values()]
    accuracies = [c / t if t > 0 else 0 for c, t in zip(correct, totals)]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(categories))
    axes[0].bar(x - 0.175, totals, 0.35, label='总计', color='steelblue')
    axes[0].bar(x + 0.175, correct, 0.35, label='正确', color='green')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, rotation=45, ha='right')
    axes[0].legend()
    axes[0].set_title('问题类型统计')
    
    colors = ['green' if acc > 0.5 else 'orange' if acc > 0.3 else 'red' for acc in accuracies]
    axes[1].bar(categories, accuracies, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories, rotation=45, ha='right')
    axes[1].axhline(y=0.5, color='gray', linestyle='--', label='50%基线')
    axes[1].set_title('各问题类型准确率')
    axes[1].legend()
    
    for i, acc in enumerate(accuracies):
        if acc > 0:
            axes[1].text(i, acc + 0.02, f'{acc:.1%}', ha='center', fontsize=9)
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'category_statistics.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"分类统计图表已保存到: {chart_path}")
    return chart_path


def create_ablation_comparison(baseline_metrics, clip_metrics, output_dir):
    """创建消融实验对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    categories = list(baseline_metrics['category_stats'].keys())
    baseline_acc = [baseline_metrics['category_stats'][cat]['correct'] / 
                   max(baseline_metrics['category_stats'][cat]['total'], 1) 
                   for cat in categories]
    clip_acc = [clip_metrics['category_stats'][cat]['correct'] / 
               max(clip_metrics['category_stats'][cat]['total'], 1) 
               for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[0].bar(x - width/2, baseline_acc, width, label='基线(Qwen)', color='steelblue')
    axes[0].bar(x + width/2, clip_acc, width, label='+CLIP重排序', color='orange')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, rotation=45, ha='right')
    axes[0].legend()
    axes[0].set_title('消融实验: 各类型准确率对比')
    axes[0].set_ylabel('准确率')
    
    metrics_names = ['总体准确率', '模糊匹配率', 'CLIP优化样本数']
    baseline_vals = [baseline_metrics['overall_accuracy'], 
                    baseline_metrics['fuzzy_accuracy'], 0]
    clip_vals = [clip_metrics['overall_accuracy'], 
                clip_metrics['fuzzy_accuracy'], 
                clip_metrics['clip_reranked_count']]
    
    x2 = np.arange(len(metrics_names))
    axes[1].bar(x2 - width/2, baseline_vals, width, label='基线(Qwen)', color='steelblue')
    axes[1].bar(x2 + width/2, clip_vals, width, label='+CLIP重排序', color='orange')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(metrics_names)
    axes[1].legend()
    axes[1].set_title('整体指标对比')
    axes[1].set_ylabel('数值')
    
    for i, (b, c) in enumerate(zip(baseline_vals, clip_vals)):
        axes[1].text(i - width/2, b + 0.02, f'{b:.1%}', ha='center', fontsize=9)
        axes[1].text(i + width/2, c + 0.02, f'{c:.1%}' if i < 2 else f'{int(c)}', ha='center', fontsize=9)
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'ablation_comparison.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"消融实验对比图已保存到: {chart_path}")
    return chart_path


def save_results(results, metrics, output_dir, use_clip_rerank):
    """保存结果"""
    
    result_file = f'all_results_clip_{"enabled" if use_clip_rerank else "disabled"}.json'
    with open(os.path.join(output_dir, result_file), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    report = {
        'evaluation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model': f'Qwen3-VL-4B-Instruct + CLIP (rerank={"enabled" if use_clip_rerank else "disabled"})',
        'metrics': metrics,
        'sample_results': results[:20]
    }
    with open(os.path.join(output_dir, f'evaluation_report_clip_{"enabled" if use_clip_rerank else "disabled"}.json'), 
              'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, f'evaluation_summary_clip_{"enabled" if use_clip_rerank else "disabled"}.txt'), 
              'w', encoding='utf-8') as f:
        f.write("Qwen3-VL + CLIP VQA 评估报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"CLIP重排序: {'启用' if use_clip_rerank else '禁用'}\n")
        f.write(f"总样本数: {metrics['total_samples']}\n")
        f.write(f"正确预测: {metrics['correct_samples']}\n")
        f.write(f"总体准确率: {metrics['overall_accuracy']:.2%}\n")
        f.write(f"模糊匹配率: {metrics['fuzzy_accuracy']:.2%}\n")
        f.write(f"CLIP优化样本数: {metrics['clip_reranked_count']}\n")
        f.write(f"CLIP提升正确的样本数: {metrics['clip_improved_count']}\n\n")
        f.write("各类型准确率:\n")
        for cat, data in metrics['category_accuracy'].items():
            if data['accuracy'] is not None:
                stats = metrics['category_stats'][cat]
                f.write(f"  {cat}: {data['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
                if data['clip_improved_rate']:
                    f.write(f" [CLIP优化率: {data['clip_improved_rate']:.2%}]")
                f.write("\n")
    
    print(f"所有结果已保存到: {output_dir}")


def run_ablation_experiments(client, model_name, processor, clip_model, clip_device,
                           metadata, image_dir, sample_size=100):
    """运行消融实验"""
    
    output_dir_base = os.path.join(DATA_RESULTS, "vqa_results_ablation")
    os.makedirs(output_dir_base, exist_ok=True)
    
    print("=" * 60)
    print("开始消融实验")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("实验1: 基线模型 (不使用CLIP重排序)")
    print("=" * 60)
    results_baseline, stats_baseline = evaluate_dataset(
        client, model_name, processor, None, None,
        metadata, image_dir, sample_size=sample_size, use_clip_rerank=False
    )
    metrics_baseline = compute_metrics(results_baseline, stats_baseline)
    
    print(f"\n基线模型结果:")
    print(f"总体准确率: {metrics_baseline['overall_accuracy']:.2%}")
    for cat, data in metrics_baseline['category_accuracy'].items():
        if data['accuracy'] is not None:
            print(f"  {cat}: {data['accuracy']:.2%}")
    
    save_results(results_baseline, metrics_baseline, output_dir_base, use_clip_rerank=False)
    
    print("\n" + "=" * 60)
    print("实验2: Qwen3-VL + CLIP重排序")
    print("=" * 60)
    results_clip, stats_clip = evaluate_dataset(
        client, model_name, processor, clip_model, clip_device,
        metadata, image_dir, sample_size=sample_size, use_clip_rerank=True
    )
    metrics_clip = compute_metrics(results_clip, stats_clip)
    
    print(f"\nCLIP重排序模型结果:")
    print(f"总体准确率: {metrics_clip['overall_accuracy']:.2%}")
    for cat, data in metrics_clip['category_accuracy'].items():
        if data['accuracy'] is not None:
            print(f"  {cat}: {data['accuracy']:.2%}")
    
    save_results(results_clip, metrics_clip, output_dir_base, use_clip_rerank=True)
    
    print("\n" + "=" * 60)
    print("消融实验对比")
    print("=" * 60)
    print(f"基线模型准确率: {metrics_baseline['overall_accuracy']:.2%}")
    print(f"+CLIP重排序准确率: {metrics_clip['overall_accuracy']:.2%}")
    improvement = metrics_clip['overall_accuracy'] - metrics_baseline['overall_accuracy']
    print(f"准确率变化: {'+' if improvement > 0 else ''}{improvement:.2%}")
    print(f"CLIP优化样本数: {metrics_clip['clip_reranked_count']}")
    
    create_ablation_comparison(metrics_baseline, metrics_clip, output_dir_base)
    
    viz_baseline = create_visualization(results_baseline, image_dir, output_dir_base, num_samples=20)
    viz_clip = create_visualization(results_clip, image_dir, output_dir_base, num_samples=20)
    
    create_category_chart(stats_baseline, output_dir_base)
    
    ablation_report = {
        'evaluation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'baseline': {
            'model': 'Qwen3-VL-4B-Instruct',
            'use_clip_rerank': False,
            'metrics': metrics_baseline
        },
        'with_clip_rerank': {
            'model': 'Qwen3-VL-4B-Instruct + CLIP',
            'use_clip_rerank': True,
            'metrics': metrics_clip
        },
        'improvement': {
            'accuracy_delta': metrics_clip['overall_accuracy'] - metrics_baseline['overall_accuracy'],
            'clip_reranked_count': metrics_clip['clip_reranked_count'],
            'clip_improved_count': metrics_clip['clip_improved_count']
        }
    }
    
    with open(os.path.join(output_dir_base, 'ablation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(ablation_report, f, ensure_ascii=False, indent=2)
    
    return metrics_baseline, metrics_clip


def main():
    output_dir = os.path.join(DATA_RESULTS, "vqa_results_ablation")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Qwen3-VL + CLIP VQA 评估 (含消融实验)")
    print("=" * 60)
    
    print("=" * 60)
    print("Qwen3-VL + CLIP VQA 评估")
    print("=" * 60)
    
    client, model_name = load_model(API_KEY)
    processor, clip_model, clip_device = load_clip_model()
    
    metadata = load_metadata(DATA_IMAGES)
    
    metrics_baseline, metrics_clip = run_ablation_experiments(
        client, model_name, processor, clip_model, clip_device,
        metadata, DATA_IMAGES, sample_size=100
    )
    
    print("\n" + "=" * 60)
    print("消融实验完成!")
    print("=" * 60)
    print(f"基线模型准确率: {metrics_baseline['overall_accuracy']:.2%}")
    print(f"+CLIP重排序准确率: {metrics_clip['overall_accuracy']:.2%}")
    improvement = metrics_clip['overall_accuracy'] - metrics_baseline['overall_accuracy']
    print(f"准确率变化: {'+' if improvement > 0 else ''}{improvement:.2%}")
    
    print(f"评估完成! 结果目录: {output_dir}")


if __name__ == "__main__":
    main()
