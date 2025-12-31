"""
VQA 公共功能模块 (LangChain版本)
================================

本文件包含VQA评估系统的公共功能模块，使用LangChain简化API调用。
主要功能包括：

1. 模型初始化和客户端配置 (LangChain)
2. 数据加载和预处理
3. 文本处理和标准化
4. 问题分类系统
5. VQA推理引擎 (LangChain简化版)
6. 评估指标计算
7. 可视化和报告生成

设计原则：
- 使用LangChain统一API调用接口
- 模块化：每个函数负责单一功能
- 可复用：支持在多个脚本中共享使用
"""

import json
import os
import re
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from io import BytesIO
import base64

plt.switch_backend('Agg')

def find_chinese_font():
    """查找可用的中文字体"""
    import matplotlib.font_manager as fm
    from pathlib import Path

    # 系统字体路径
    font_dirs = [
        Path("C:/Windows/Fonts"),
        Path("/usr/share/fonts"),
        Path.home() / ".fonts"
    ]

    # 纯中文字体列表（按优先级）
    chinese_font_names = [
        'Microsoft YaHei', 'MicrosoftYaHei', 'msyh',
        'SimHei', 'simhei', 'hei',
        'Noto Sans CJK SC', 'NotoSansCJKsc',
        'WenQuanYi Micro Hei', 'wqy-microhei',
        'Droid Sans Fallback', 'droidfallback',
        'PingFang SC', 'PingFang', 'pingfang',
        'Heiti SC', 'heiti',
        'Source Han Sans CN', 'sourcehansanscn',
        'AR PL UMing CN', 'umingcn',
        'AR PL Sungti CN', 'sungti',
    ]

    # 查找系统中可用的字体文件
    font_files = []
    for font_dir in font_dirs:
        if font_dir.exists():
            for ext in ['.ttf', '.ttc', '.otf']:
                font_files.extend(font_dir.glob(f'*{ext}'))

    # 检测每个字体文件
    for font_file in font_files:
        try:
            font_name = font_file.stem.lower()
            for cn_font in chinese_font_names:
                if cn_font.lower() in font_name:
                    return str(font_file)
        except:
            continue

    # 如果没找到，返回None
    return None

# 尝试加载中文字体
selected_font_path = find_chinese_font()

if selected_font_path:
    try:
        fm.fontManager.addfont(selected_font_path)
        font_prop = fm.FontProperties(fname=selected_font_path)
        font_name = fm.FontProperties(fname=selected_font_path).get_name()
        plt.rcParams['font.family'] = font_name
        plt.rcParams['font.sans-serif'] = [font_name]
        print(f"使用中文字体: {font_name} ({selected_font_path})")
    except Exception as e:
        print(f"字体加载失败: {e}")
        selected_font_path = None

# 如果无法加载中文字体，使用默认设置
if not selected_font_path:
    print("未找到合适的中文字体，将使用系统默认")
    plt.rcParams['axes.unicode_minus'] = False

# 设置默认字体属性函数（用于中文文本）
def get_chinese_font():
    """获取中文字体属性"""
    if selected_font_path:
        return fm.FontProperties(fname=selected_font_path)
    return None


try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.tools import tool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain未安装，将使用原始OpenAI API")


def image_to_base64(image_path):
    """将图像转换为base64编码"""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            max_size = 1024
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"图片处理错误 {image_path}: {e}")
        return None


def load_model(api_key, model_name="qwen3-vl-8b-instruct"):
    """
    初始化Qwen3-VL客户端 (LangChain或原生API)

    Args:
        api_key (str): DashScope API密钥
        model_name (str): 模型名称

    Returns:
        tuple: (llm, model_name) - LangChain LLM对象和模型名称
    """
    print("初始化 Qwen3-VL 客户端...")

    if LANGCHAIN_AVAILABLE:
        llm = ChatOpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model=model_name,
            temperature=0.1,
            max_tokens=128
        )
        print(f"LangChain客户端已创建，使用模型: {model_name}")
    else:
        from openai import OpenAI
        llm = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        print(f"原生OpenAI客户端已创建，使用模型: {model_name}")

    return llm, model_name


def vqa_inference(llm, model_name, image_path, question):
    """
    视觉问答推理函数 (LangChain简化版)

    使用LangChain的标准化接口进行VQA推理。

    Args:
        llm: LangChain LLM客户端或原生OpenAI客户端
        model_name (str): 模型名称
        image_path (str): 图像路径
        question (str): 问题文本

    Returns:
        str: 模型回答或错误信息
    """
    try:
        base64_image = image_to_base64(image_path)
        if not base64_image:
            return "错误: 无法处理图片"

        full_question = f"{question}\n请简短回答，只回答关键信息，不需要解释。"

        if LANGCHAIN_AVAILABLE:
            message = HumanMessage(
                content=[
                    {"type": "text", "text": full_question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            )
            response = llm.invoke([message])
            return response.content.strip()
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": full_question}
                    ]
                }
            ]
            response = llm.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=128,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"API调用错误: {e}")
        return f"错误: {str(e)}"


def load_metadata(data_dir):
    """加载数据集元数据"""
    metadata_path = os.path.join(data_dir, "metadata.json")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"加载了 {len(metadata)} 条元数据")
    return metadata


def normalize_answer(answer):
    """答案标准化处理"""
    if not answer:
        return ""
    answer = answer.lower().strip()
    answer = re.sub(r'[^\w\s]', ' ', answer)
    answer = re.sub(r'\s+', ' ', answer)
    return answer.strip()


def classify_question(question):
    """问题类型自动分类"""
    q = question.lower()

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

    for cat, keywords in categories.items():
        if any(kw in q for kw in keywords):
            return cat
    return 'other'


def compute_exact_match(pred, targets):
    """精确匹配评估"""
    pred_norm = normalize_answer(pred)
    for target in targets:
        if pred_norm == normalize_answer(target):
            return True
    return False


def compute_fuzzy_match(pred, targets):
    """模糊匹配算法"""
    pred_norm = normalize_answer(pred)

    for target in targets:
        target_norm = normalize_answer(target)

        if pred_norm == target_norm:
            return True
        if target_norm in pred_norm:
            return True
        if pred_norm in target_norm and len(pred_norm) > 3:
            pred_words = set(pred_norm.split())
            target_words = set(target_norm.split())
            if target_words:
                overlap_ratio = len(pred_words & target_words) / len(target_words)
                if overlap_ratio > 0.7:
                    return True
    return False


def compute_accuracy(pred, targets):
    """计算预测答案的准确率"""
    pred_norm = normalize_answer(pred)

    for target in targets:
        target_norm = normalize_answer(target)
        if pred_norm == target_norm:
            return True

    best_score = 0
    for target in targets:
        target_norm = normalize_answer(target)
        score = 0
        if target_norm in pred_norm:
            score = 0.9
        elif pred_norm in target_norm:
            score = 0.8
        else:
            common = set(pred_norm.split()) & set(target_norm.split())
            if target_norm.split():
                score = len(common) / len(set(target_norm.split()))
        if score > best_score:
            best_score = score

    return best_score >= 0.6


def create_category_chart(category_stats, output_dir):
    """创建问题类型统计图表"""
    categories = list(category_stats.keys())
    totals = [stats['total'] for stats in category_stats.values()]
    correct = [stats['correct'] for stats in category_stats.values()]
    accuracies = [c / t if t > 0 else 0 for c, t in zip(correct, totals)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(categories))
    ax1.bar(x - 0.175, totals, 0.35, label='总计', color='steelblue')
    ax1.bar(x + 0.175, correct, 0.35, label='正确', color='green')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend()
    ax1.set_title('问题类型统计')

    colors = ['green' if acc > 0.5 else 'orange' if acc > 0.3 else 'red' for acc in accuracies]
    ax2.bar(categories, accuracies, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.axhline(y=0.5, color='gray', linestyle='--', label='50%基线')
    ax2.set_title('各问题类型准确率')
    ax2.legend()

    for i, acc in enumerate(accuracies):
        if acc > 0:
            ax2.text(i, acc + 0.02, f'{acc:.1%}', ha='center', fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'category_statistics.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"分类统计图表已保存到: {chart_path}")
    return chart_path


def create_visualization(results, image_dir, output_dir, num_samples=20, show_clip_info=False):
    """创建VQA评估结果的可视化展示"""
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

            title_parts = [
                f"ID:{result['id']} [{result['category']}] {status}",
                f"问题: {result['question'][:35]}..."
            ]

            if show_clip_info and 'clip_score' in result:
                answer_field = 'final_answer' if 'final_answer' in result else 'model_answer'
                clip_info = f"\nCLIP得分: {result['clip_score']:.2f}" if result['clip_score'] > 0 else ""
                rerank_info = " (CLIP优化)" if result.get('clip_reranked', False) else ""

                title_parts.append(
                    f"预测: {result[answer_field][:25]}{rerank_info}{clip_info}"
                )
                if 'ground_truth' in result:
                    title_parts.append(
                        f"真实: {result['ground_truth'][:25]}"
                    )
            else:
                answer_field = 'model_answer' if 'model_answer' in result else 'final_answer'
                title_parts.append(
                    f"答案: {result[answer_field][:30]}"
                )

            ax.set_title(
                "\n".join(title_parts),
                fontsize=8, color=color
            )
        except Exception as e:
            ax.text(0.5, 0.5, f"图像加载失败", ha='center', va='center')
            ax.axis('off')

    for ax in axes.flat[len(selected):]:
        ax.axis('off')

    plt.tight_layout()
    viz_path = os.path.join(output_dir, 'vqa_visualization.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化结果已保存到: {viz_path}")
    return viz_path


def compute_metrics_base(results, category_stats):
    """计算基础评估指标"""
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    overall_accuracy = correct / total if total > 0 else 0

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


def save_classification_result(image_path, labels, all_probs, output_dir):
    """保存分类结果到JSON文件"""
    result = {
        "image": str(image_path),
        "top_label": labels[all_probs.argmax()],
        "top_prob": float(all_probs.max()),
        "all_predictions": {label: float(prob) for label, prob in zip(labels, all_probs)}
    }
    output_path = os.path.join(output_dir, "classification_result.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"分类结果已保存至: {output_path}")
    return output_path


def save_retrieval_result(query_text, results, output_dir):
    """保存检索结果到JSON文件"""
    result = {
        "query": query_text,
        "results": results
    }
    output_path = os.path.join(output_dir, "retrieval_result.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"检索结果已保存至: {output_path}")
    return output_path


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


def get_pil_font(size):
    """获取适合的PIL字体（支持中文）"""
    font_candidates = []
    if selected_font_path:
        font_candidates.append(selected_font_path)
    font_candidates.extend([
        "arial.ttf", "Arial.ttf",
        "simhei.ttf", "SimHei.ttf", "simsun.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ])
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except:
            continue
    return ImageFont.load_default()

def create_visualization_comparison(image_path, labels, probs, output_path, task_type="classification"):
    """创建原图与输出结果的对比可视化图像"""
    original_image = Image.open(image_path).convert("RGB")
    width, height = original_image.size
    result_width = width * 2
    result_height = height
    result_image = Image.new('RGB', (result_width, result_height), (255, 255, 255))
    result_image.paste(original_image, (0, 0))
    draw = ImageDraw.Draw(result_image)
    font_size = int(min(width, height) * 0.035)
    title_font = get_pil_font(font_size)
    label_font = get_pil_font(int(font_size * 0.8))
    prob_font = get_pil_font(int(font_size * 0.7))
    info_x = width + 20
    info_y = 20
    draw.text((info_x, info_y), "CLIP Zero-Shot 分类结果", fill=(0, 0, 0), font=title_font)
    info_y += font_size + 30
    sorted_indices = np.argsort(probs)[::-1]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_probs = [probs[i] for i in sorted_indices]
    bar_start_x = info_x
    bar_start_y = info_y + font_size
    bar_height = int(height * 0.7 / len(labels))
    max_bar_width = int(width * 0.35)
    max_prob = max(sorted_probs) if max(sorted_probs) > 0 else 1.0
    for i, (label, prob) in enumerate(zip(sorted_labels, sorted_probs)):
        label_y = bar_start_y + i * bar_height
        bar_width = int((prob / max_prob) * max_bar_width) if max_prob > 0 else 0
        bar_color = (int(50 + 200 * (1 - i / len(labels))),
                     int(100 + 155 * (i / len(labels))),
                     255)
        draw.rectangle([bar_start_x, label_y, bar_start_x + bar_width, label_y + bar_height - 2],
                      fill=bar_color)
        draw.text((bar_start_x + bar_width + 5, label_y),
                 f"{prob:.3f}", fill=(0, 0, 0), font=prob_font)
    info_y = bar_start_y + len(labels) * bar_height + 20
    result_image.save(output_path, quality=95)
    print(f"可视化对比图已保存至: {output_path}")


def create_retrieval_comparison(query_text, results, image_folder, output_path):
    """创建图文检索结果对比可视化"""
    if not results:
        return
    sample_image = Image.open(results[0]["image"]).convert("RGB")
    img_width, img_height = sample_image.size
    result_count = len(results)
    total_width = img_width * (result_count + 1)
    total_height = img_height
    result_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(result_image)
    font_size = int(min(img_width, img_height) * 0.04)
    title_font = get_pil_font(font_size)
    label_font = get_pil_font(int(font_size * 0.7))
    draw.text((20, 20), f"查询: '{query_text}'", fill=(0, 0, 150), font=title_font)
    for i, result in enumerate(results):
        img = Image.open(result["image"]).convert("RGB")
        x_offset = img_width * (i + 1)
        result_image.paste(img, (x_offset, 0))
        label_y = 20
        label_text = f"排名 {result['rank']}"
        draw.text((x_offset + 20, label_y), label_text, fill=(0, 0, 0), font=title_font)
        score_y = label_y + font_size + 10
        score_text = f"相似度: {result['similarity']:.3f}"
        draw.text((x_offset + 20, score_y), score_text, fill=(0, 100, 0), font=label_font)
    result_image.save(output_path, quality=95)
    print(f"检索结果对比图已保存至: {output_path}")


def create_vqa_visualization_comparison(image_path, question, answer, output_path):
    """创建原图与VQA输出结果的对比可视化图像"""
    original_image = Image.open(image_path).convert("RGB")
    width, height = original_image.size
    dpi = 100
    fig_width = (width * 2) / dpi
    fig_height = height / dpi
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    axes[0].imshow(original_image)
    axes[0].set_title("原图", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    font_prop = get_chinese_font()
    info_text = f"问题: {question}\n\n答案: {answer}"
    axes[1].text(0.05, 0.95, info_text,
                 transform=axes[1].transAxes, fontsize=11,
                 verticalalignment='top', fontproperties=font_prop,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    axes[1].set_title("Qwen3-VL VQA 结果", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"可视化对比图已保存至: {output_path}")
