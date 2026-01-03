"""
================================================================================
VQA 公共功能模块
================================================================================
功能概览：
1. 模型初始化和客户端配置
2. 数据加载和预处理
3. 文本处理和标准化
4. VQA推理引擎
5. 评估指标计算
6. 可视化和报告生成
================================================================================
"""

import json
import os
import re
import base64
from io import BytesIO
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.switch_backend('Agg')

# ==============================================================================
# 字体配置
# ==============================================================================

def _find_font():
    """查找可用的中文字体"""
    from pathlib import Path
    font_names = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi', 'PingFang SC']
    for dir_path in [Path("C:/Windows/Fonts"), Path("/usr/share/fonts")]:
        if dir_path.exists():
            font_files = list(dir_path.glob('*.ttf')) + list(dir_path.glob('*.ttc'))
            for font_path in font_files:
                for name in font_names:
                    if name.lower() in font_path.stem.lower():
                        return str(font_path)
    return None

selected_font_path = _find_font()

if selected_font_path:
    try:
        import matplotlib.font_manager as fm
        fm.fontManager.addfont(selected_font_path)
        font_name = fm.FontProperties(fname=selected_font_path).get_name()
        plt.rcParams['font.family'] = font_name
        print(f"使用中文字体: {font_name}")
    except:
        selected_font_path = None
else:
    plt.rcParams['axes.unicode_minus'] = False


def get_chinese_font():
    """获取中文字体对象"""
    if selected_font_path:
        return fm.FontProperties(fname=selected_font_path)
    return None


# ==============================================================================
# 全局LLM配置
# ==============================================================================

_llm_config = {}


def _init_llm_config(llm, api_key, base_url):
    """初始化LLM配置"""
    global _llm_config
    _llm_config = {
        'api_key': api_key,
        'base_url': base_url
    }


def _get_api_key():
    """获取API密钥"""
    return _llm_config.get('api_key', '')


def _get_base_url():
    """获取Base URL"""
    return _llm_config.get('base_url', '')


# ==============================================================================
# 图像处理函数
# ==============================================================================
def image_to_base64(image_path):
    """将图像转换为base64编码"""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            max_size = 1024
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                img = img.resize(tuple(int(d * ratio) for d in img.size), Image.LANCZOS)
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"图片处理错误: {e}")
        return None


# ==============================================================================
# 第五部分：模型初始化函数
# ==============================================================================
def load_model(api_key, model_name="qwen3-vl-8b-instruct"):
    """初始化Qwen3-VL模型客户端"""
    from langchain_openai import ChatOpenAI

    print("初始化 Qwen3-VL 客户端...")

    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model=model_name,
        temperature=0.1,
        max_tokens=128
    )

    _init_llm_config(llm, api_key, "https://dashscope.aliyuncs.com/compatible-mode/v1")
    print(f"LangChain客户端已创建，使用模型: {model_name}")

    return llm, model_name


# ==============================================================================
# 第六部分：VQA推理函数
# ==============================================================================
def vqa_inference_with_temperature(llm, model_name, image_path, question, temperature=0.7, custom_prompt=None):
    """带温度参数的VQA推理函数"""
    try:
        from langchain_openai import ChatOpenAI

        base64_image = image_to_base64(image_path)
        if not base64_image:
            return "错误: 无法处理图片"

        full_question = custom_prompt if custom_prompt else f"{question}\n请简短回答，只回答关键信息，不需要解释。"

        temp_llm = ChatOpenAI(
            api_key=_get_api_key(),
            base_url=_get_base_url(),
            model=model_name,
            temperature=temperature,
            max_tokens=128
        )

        message = HumanMessage(
            content=[
                {"type": "text", "text": full_question},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )

        response = temp_llm.invoke([message])
        return response.content.strip()

    except Exception as e:
        print(f"API调用错误: {e}")
        return f"错误: {str(e)}"


def generate_multiple_candidates(llm, model_name, image_path, question, num_candidates=5):
    """
    生成多个候选答案
    """
    answers = []
    concise_prompt = f"{question}\n请用最简短的词或短语回答，1-3个词足矣，不要任何解释。"

    for i in range(num_candidates):
        temp = 0.7 + i * 0.1
        answer = vqa_inference_with_temperature(llm, model_name, image_path, question, temperature=temp, custom_prompt=concise_prompt)
        if answer and not answer.startswith("错误:"):
            answers.append(answer)

    unique_answers = list(set(answers))
    return unique_answers


# ==============================================================================
# 第七部分：数据加载函数
# ==============================================================================
def load_metadata(data_dir):
    """
    加载数据集元数据
    """
    # 构建metadata.json文件的完整路径
    metadata_path = os.path.join(data_dir, "metadata.json")
    
    # 打开并解析JSON文件
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    print(f"加载了 {len(metadata)} 条元数据")
    return metadata


# ==============================================================================
# 第八部分：文本处理函数
# ==============================================================================
def normalize_answer(answer):
    """
    答案标准化处理
    """
    if not answer:
        return ""
        
    # 转换为小写并去除首尾空白
    answer = answer.lower().strip()
    
    # 移除标点符号（保留字母、数字和空格）
    # [^\w\s] 匹配非字母数字下划线和空白的字符
    answer = re.sub(r'[^\w\s]', ' ', answer)
    
    # 合并多个连续空格为单个空格
    answer = re.sub(r'\s+', ' ', answer)
    
    return answer.strip()


# ==============================================================================
# 第八部分：评估指标计算函数
# ==============================================================================
def compute_match(pred, targets):
    """
    综合匹配评估
    """
    pred_norm = normalize_answer(pred)

    for target in targets:
        target_norm = normalize_answer(target)

        if pred_norm == target_norm:
            return True
            
        if target_norm in pred_norm:
            return True
            
        pred_words = set(pred_norm.split())
        target_words = set(target_norm.split())
        
        if target_words:
            overlap_ratio = len(pred_words & target_words) / len(target_words)
            if overlap_ratio > 0.7:
                return True
                    
    return False


def compute_accuracy(pred, targets):
    """
    综合准确率计算
    """
    # 标准化预测答案
    pred_norm = normalize_answer(pred)

    for target in targets:
        target_norm = normalize_answer(target)
        if pred_norm == target_norm:
            return True

    # 计算最佳匹配分数
    best_score = 0
    for target in targets:
        target_norm = normalize_answer(target)
        score = 0
        
        if target_norm in pred_norm:
            score = 0.9
        elif pred_norm in target_norm:
            score = 0.8
        else:
            # 计算词汇重叠度
            common = set(pred_norm.split()) & set(target_norm.split())
            if target_norm.split():
                score = len(common) / len(set(target_norm.split()))
                
        if score > best_score:
            best_score = score

    # 阈值判断
    return best_score >= 0.6


# ==============================================================================
# 第十部分：可视化函数
# ==============================================================================
def create_category_chart(results, output_dir):
    """
    创建VQA评估结果统计图表
    """
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    accuracy = correct / total if total > 0 else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(['总计', '正确'], [total, correct], color=['steelblue', 'green'])
    ax1.set_title('样本统计')
    ax1.set_ylabel('数量')
    for i, v in enumerate([total, correct]):
        ax1.text(i, v + 1, str(v), ha='center')

    ax2.bar(['准确率'], [accuracy], color=['steelblue'])
    ax2.set_title('总体准确率')
    ax2.set_ylabel('准确率')
    ax2.set_ylim(0, 1)
    ax2.axhline(y=0.5, color='gray', linestyle='--', label='50%基线')
    ax2.text(0, accuracy + 0.02, f'{accuracy:.1%}', ha='center')

    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'category_statistics.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"统计图表已保存到: {chart_path}")
    return chart_path


def create_visualization(results, image_dir, output_dir, num_samples=20, show_clip_info=False):
    """
    创建VQA评估结果的可视化展示
    """
    # 分离正确和错误预测
    success = [r for r in results if r['is_correct']]
    failure = [r for r in results if not r['is_correct']]

    # 选择展示样本
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

    # 随机打乱顺序
    np.random.shuffle(selected)

    # 计算网格布局
    cols, rows = 4, (len(selected) + 3) // 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()

    # 绘制每个样本
    for idx, result in enumerate(selected):
        if idx >= len(axes):
            break
        ax = axes[idx]
        try:
            # 读取并显示图像
            img = Image.open(os.path.join(image_dir, result['image_file']))
            ax.imshow(img)
            ax.axis('off')

            # 确定状态颜色和标签
            status = "[成功]" if result['is_correct'] else "[失败]"
            color = 'green' if result['is_correct'] else 'red'

            # 构建标题文本
            title_parts = [
                f"ID:{result['id']} {status}",
                f"问题: {result['question'][:35]}..."
            ]

            # 如果显示CLIP信息
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

    # 隐藏多余的子图
    for ax in axes.flat[len(selected):]:
        ax.axis('off')

    # 保存图表
    plt.tight_layout()
    viz_path = os.path.join(output_dir, 'vqa_visualization.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化结果已保存到: {viz_path}")
    return viz_path


def get_pil_font(size):
    """
    获取适合的PIL字体（支持中文）
    """
    font_candidates = []
    
    # 优先使用检测到的中文字体
    if selected_font_path:
        font_candidates.append(selected_font_path)
        
    # 添加其他候选字体
    font_candidates.extend([
        "arial.ttf", "Arial.ttf",
        "simhei.ttf", "SimHei.ttf", "simsun.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ])
    
    # 尝试加载每个候选字体
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except:
            continue
            
    # 都失败则返回默认字体
    return ImageFont.load_default()


def create_visualization_comparison(image_path, labels, probs, output_path, task_type="classification"):
    """
    创建原图与输出结果的对比可视化图像
    """
    # 打开原图
    original_image = Image.open(image_path).convert("RGB")
    width, height = original_image.size
    
    # 创建输出图像（2倍宽度）
    result_width = width * 2
    result_height = height
    result_image = Image.new('RGB', (result_width, result_height), (255, 255, 255))
    result_image.paste(original_image, (0, 0))
    
    draw = ImageDraw.Draw(result_image)
    
    # 计算字体大小
    font_size = int(min(width, height) * 0.035)
    title_font = get_pil_font(font_size)
    label_font = get_pil_font(int(font_size * 0.8))
    prob_font = get_pil_font(int(font_size * 0.7))
    
    # 绘制标题
    info_x = width + 20
    info_y = 20
    draw.text((info_x, info_y), "CLIP Zero-Shot 分类结果", fill=(0, 0, 0), font=title_font)
    info_y += font_size + 30
    
    # 按概率排序
    sorted_indices = np.argsort(probs)[::-1]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_probs = [probs[i] for i in sorted_indices]
    
    # 绘制概率条形图
    bar_start_x = info_x
    bar_start_y = info_y + font_size
    bar_height = int(height * 0.7 / len(labels))
    max_bar_width = int(width * 0.35)
    max_prob = max(sorted_probs) if max(sorted_probs) > 0 else 1.0
    
    for i, (label, prob) in enumerate(zip(sorted_labels, sorted_probs)):
        label_y = bar_start_y + i * bar_height
        bar_width = int((prob / max_prob) * max_bar_width) if max_prob > 0 else 0
        
        # 颜色渐变（高概率为绿色，低概率为蓝色）
        bar_color = (int(50 + 200 * (1 - i / len(labels))),
                     int(100 + 155 * (i / len(labels))),
                     255)
        
        # 绘制条形
        draw.rectangle([bar_start_x, label_y, bar_start_x + bar_width, label_y + bar_height - 2],
                      fill=bar_color)
        
        # 绘制概率数值
        draw.text((bar_start_x + bar_width + 5, label_y),
                 f"{prob:.3f}", fill=(0, 0, 0), font=prob_font)
        
        # 绘制标签（如果有空间）
        label_text_y = bar_start_y + i * bar_height + bar_height // 4
        draw.text((bar_start_x, label_text_y), label, fill=(0, 0, 0), font=label_font)
        
    # 保存结果
    result_image.save(output_path, quality=95)
    print(f"可视化对比图已保存至: {output_path}")


def create_retrieval_comparison(query_text, results, image_folder, output_path):
    """
    创建图文检索结果对比可视化
    """
    if not results:
        return
        
    # 获取样本图像尺寸
    sample_image = Image.open(results[0]["image"]).convert("RGB")
    img_width, img_height = sample_image.size
    
    # 创建结果图像
    result_count = len(results)
    total_width = img_width * (result_count + 1)
    total_height = img_height
    result_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(result_image)
    
    # 计算字体大小
    font_size = int(min(img_width, img_height) * 0.04)
    title_font = get_pil_font(font_size)
    label_font = get_pil_font(int(font_size * 0.7))
    
    # 绘制查询文本
    draw.text((20, 20), f"查询: '{query_text}'", fill=(0, 0, 150), font=title_font)
    
    # 拼接每个检索结果
    for i, result in enumerate(results):
        img = Image.open(result["image"]).convert("RGB")
        x_offset = img_width * (i + 1)
        result_image.paste(img, (x_offset, 0))
        
        # 绘制排名和相似度
        label_y = 20
        label_text = f"排名 {result['rank']}"
        draw.text((x_offset + 20, label_y), label_text, fill=(0, 0, 0), font=title_font)
        
        score_y = label_y + font_size + 10
        score_text = f"相似度: {result['similarity']:.3f}"
        draw.text((x_offset + 20, score_y), score_text, fill=(0, 100, 0), font=label_font)
        
    # 保存结果
    result_image.save(output_path, quality=95)
    print(f"检索结果对比图已保存至: {output_path}")


def create_vqa_visualization_comparison(image_path, question, answer, output_path):
    """
    创建原图与VQA输出结果的对比可视化图像
    """
    # 打开原图
    original_image = Image.open(image_path).convert("RGB")
    width, height = original_image.size
    
    # 计算输出图像尺寸
    dpi = 100
    fig_width = (width * 2) / dpi
    fig_height = height / dpi
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    
    # 左侧：原图
    axes[0].imshow(original_image)
    axes[0].set_title("原图", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 右侧：问答结果
    font_prop = get_chinese_font()
    info_text = f"问题: {question}\n\n答案: {answer}"
    axes[1].text(0.05, 0.95, info_text,
                 transform=axes[1].transAxes, fontsize=11,
                 verticalalignment='top', fontproperties=font_prop,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    axes[1].set_title("Qwen3-VL VQA 结果", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"可视化对比图已保存至: {output_path}")


# ==============================================================================
# 第十一部分：评估指标计算函数
# ==============================================================================
def compute_metrics_base(results):
    """
    计算基础评估指标
    """
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    overall_accuracy = correct / total if total > 0 else 0

    return {
        'overall_accuracy': overall_accuracy,
        'total_samples': total,
        'correct_samples': correct
    }


# ==============================================================================
# 第十二部分：结果保存函数
# ==============================================================================
def save_classification_result(image_path, labels, all_probs, output_dir):
    """
    保存分类结果到JSON文件
    """
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
    """
    保存检索结果到JSON文件
    """
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
    """
    保存VQA结果到JSON文件
    """
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