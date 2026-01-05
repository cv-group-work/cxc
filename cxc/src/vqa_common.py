"""
================================================================================
VQA 公共功能模块
================================================================================

本文件是VQA（Visual Question Answering，视觉问答）评估系统的核心公共功能模块，

功能概览：
1. 模型初始化和客户端配置 - 支持LangChain调用方式
2. 数据加载和预处理 - 加载数据集元数据，处理图像文件
3. 文本处理和标准化 - 答案标准化、文本清理
4. 问题分类系统 - 自动识别问题类型（计数、属性、空间关系等）
5. VQA推理引擎 - 调用视觉语言模型进行问答
6. 评估指标计算 - 精确匹配、模糊匹配、准确率计算
7. 可视化和报告生成 - 生成评估结果图表和统计报告

================================================================================
"""

# ==============================================================================
# 第一部分：导入必要的库
# ==============================================================================

# 标准库导入
import json
import os
import re
import base64
from io import BytesIO
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 图像处理库
from PIL import Image, ImageDraw, ImageFont

# 数值计算和科学计算库
import numpy as np

# 数据可视化库
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置matplotlib后端为Agg（无GUI模式），适用于服务器环境
plt.switch_backend('Agg')

# ==============================================================================
# 第二部分：中文字体配置
# ==============================================================================
# matplotlib默认不支持中文显示，需要配置中文字体
# 本模块提供自动检测和加载系统可用的中文字体功能


def find_chinese_font():
    """
    查找系统中可用的中文字体
    
    该函数遍历系统字体目录，搜索可用的中文字体文件。
    支持多种常见的中文字体，包括微软雅黑、黑体、Noto系列等。
    
    搜索策略：
    1. 优先检查Windows系统字体目录
    2. 检查Linux系统字体目录
    3. 检查用户目录下的字体
    4. 遍历字体文件，匹配已知的中文字体名称
    
    Returns:
        str or None: 找到的中文字体文件路径，如果未找到则返回None

    """
    import matplotlib.font_manager as fm
    from pathlib import Path

    # 系统字体目录列表
    # 按优先级排列，优先搜索Windows系统字体
    font_dirs = [
        Path("C:/Windows/Fonts"),           # Windows系统字体
        Path("/usr/share/fonts"),           # Linux系统字体
        Path.home() / ".fonts"              # 用户字体目录
    ]

    # 纯中文字体名称列表（不区分大小写匹配）
    # 这些是常见的中文字体变体名称
    chinese_font_names = [

        'Microsoft YaHei', 'MicrosoftYaHei', 'msyh',   # 微软雅黑
        'SimHei', 'simhei', 'hei',                     # 黑体
        'Noto Sans CJK SC', 'NotoSansCJKsc',          # Google Noto
        'WenQuanYi Micro Hei', 'wqy-microhei',        # 文泉驿
        'Droid Sans Fallback', 'droidfallback',       # Android回退字体
        'PingFang SC', 'PingFang', 'pingfang',        # 苹方（Mac）
        'Heiti SC', 'heiti',                          # 黑体（Mac）
        'Source Han Sans CN', 'sourcehansanscn',      # 思源黑体
        'AR PL UMing CN', 'umingcn',                  # 文鼎字体
        'AR PL Sungti CN', 'sungti',                  # 文鼎宋体
    ]

    # 收集所有字体文件
    font_files = []
    for font_dir in font_dirs:
        if font_dir.exists():
            # 查找所有常见字体格式的文件
            for ext in ['.ttf', '.ttc', '.otf']:
                font_files.extend(font_dir.glob(f'*{ext}'))

    # 遍历字体文件，查找中文字体
    for font_file in font_files:
        try:
            # 获取字体文件名（不含扩展名）
            font_name = font_file.stem.lower()
            for cn_font in chinese_font_names:
                if cn_font.lower() in font_name:
                    return str(font_file)
        except Exception:
            # 跳过无法读取的字体文件
            continue

    # 如果没找到任何中文字体，返回None
    return None


# 尝试加载中文字体
# 全局变量selected_font_path存储找到的字体路径
selected_font_path = find_chinese_font()

# 配置matplotlib字体设置
if selected_font_path:
    try:
        # 将字体添加到matplotlib字体管理器
        fm.fontManager.addfont(selected_font_path)
        
        # 创建字体属性对象
        font_prop = fm.FontProperties(fname=selected_font_path)
        font_name = fm.FontProperties(fname=selected_font_path).get_name()
        
        # 设置matplotlib的默认字体
        plt.rcParams['font.family'] = font_name
        plt.rcParams['font.sans-serif'] = [font_name]
        
        print(f"使用中文字体: {font_name} ({selected_font_path})")
    except Exception as e:
        print(f"字体加载失败: {e}")
        selected_font_path = None

# 如果无法加载中文字体，使用默认设置
if not selected_font_path:
    print("未找到合适的中文字体，将使用系统默认")
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False


def get_chinese_font():
    """
    获取中文字体属性对象
    
    该函数用于获取已配置的中文字体属性，
    可用于matplotlib绘图中指定中文字体。
    
    Returns:
        FontProperties or None: 中文字体属性对象，如果未找到字体则返回None

    """
    if selected_font_path:
        return fm.FontProperties(fname=selected_font_path)
    return None



# ==============================================================================
# 第四部分：图像处理函数
# ==============================================================================


def image_to_base64(image_path):
    """
    将图像转换为base64编码字符串
    
    该函数读取图像文件，进行必要的预处理（格式转换、尺寸调整），
    然后将其转换为base64编码的JPEG格式字符串。
    base64编码可用于在API调用中传输图像数据。
    
    处理步骤：
    1. 打开图像文件
    2. 转换为RGB模式（确保支持RGBA等格式）
    3. 调整图像尺寸（最大边不超过1024像素）
    4. 压缩为JPEG格式（质量85%）
    5. 转换为base64编码
    
    Args:
        image_path (str): 图像文件的路径
        
    Returns:
        str or None: base64编码的图像字符串，如果处理失败则返回None

    """
    try:
        # 打开图像文件
        with Image.open(image_path) as img:
            # 转换为RGB模式
            # 处理PNG等带透明通道的图像
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 计算缩放比例
            # 限制最大边为1024像素，减少传输数据量
            max_size = 1024
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # 创建内存缓冲区存储图像数据
            buffered = BytesIO()
            # 保存为JPEG格式，质量设置为85%（平衡质量和大小）
            img.save(buffered, format="JPEG", quality=85)
            
            # 转换为base64字符串
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
            
    except Exception as e:
        # 捕获所有异常，打印错误信息并返回None
        print(f"图片处理错误 {image_path}: {e}")
        return None


# ==============================================================================
# 第五部分：模型初始化函数
# ==============================================================================


def load_model(api_key, model_name="qwen3-vl-8b-instruct"):
    """
    初始化Qwen3-VL模型客户端
    
    该函数创建用于调用Qwen3-VL模型的客户端对象。
    调用方式：LangChain统一接口

    
    Args:
        api_key (str): DashScope API密钥
        model_name (str): 模型名称，默认为"qwen3-vl-8b-instruct"
        
    Returns:
        tuple: 包含两个元素的元组
            - llm: LangChain LLM对象
            - model_name (str): 实际使用的模型名称


    """
    print("初始化 Qwen3-VL 客户端...")


    # 使用LangChain的ChatOpenAI接口
    llm = ChatOpenAI(
        api_key=api_key,
        # 阿里云DashScope的兼容模式端点
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model=model_name,
        temperature=0.1,       # 低温度，答案更稳定
        max_tokens=128        # 最大生成128个token
    )
    print(f"LangChain客户端已创建，使用模型: {model_name}")


    return llm, model_name


# ==============================================================================
# 第六部分：VQA推理函数
# ==============================================================================


def vqa_inference(llm, model_name, image_path, question):
    """
    视觉问答推理函数
    
    该函数接收图像路径和问题文本，调用视觉语言模型生成答案。
    这是VQA系统的核心推理函数。
    
    处理流程：
    1. 将图像转换为base64编码
    2. 构造包含图像和问题的消息
    3. 调用模型API生成答案
    4. 提取并返回答案文本
    
    Args:
        llm: LangChain LLM对象或原生OpenAI客户端
        model_name (str): 模型名称
        image_path (str): 图像文件的路径
        question (str): 要询问的问题
        
    Returns:
        str: 模型生成的答案，如果出错则返回错误信息字符串

    """
    try:
        # 将图像转换为base64编码
        base64_image = image_to_base64(image_path)
        if not base64_image:
            return "错误: 无法处理图片"

        # 构造完整的问题文本
        # 要求模型简短回答关键信息
        full_question = f"{question}\n请简短回答，只回答关键信息，不需要解释。"

        # HumanMessage支持多模态内容（文本+图像）
        message = HumanMessage(
            content=[
                {"type": "text", "text": full_question},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )
        # 调用模型生成答案
        response = llm.invoke([message])
        # 提取答案内容并去除首尾空白
        return response.content.strip()


    except Exception as e:
        # 捕获API调用异常
        print(f"API调用错误: {e}")
        return f"错误: {str(e)}"


# ==============================================================================
# 第七部分：数据加载函数
# ==============================================================================


def load_metadata(data_dir):
    """
    加载数据集元数据
    
    该函数从指定目录加载metadata.json文件，
    该文件包含测试数据集的所有标注信息。
    
    数据格式：
        [
            {
                "id": "image_001",
                "image_file": "0.jpg",
                "question": "这张图片中有什么？",
                "answers": ["一只猫", "猫", "猫咪"]
            },
            ...
        ]
        
    Args:
        data_dir (str): 数据目录路径
        
    Returns:
        list: 元数据列表，每个元素是一个字典
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
    
    该函数对答案文本进行标准化处理，
    便于后续的匹配和比较操作。
    
    处理步骤：
    1. 转换为小写（消除大小写差异）
    2. 去除首尾空白
    3. 移除标点符号（替换为空格）
    4. 合并多个空格为单个空格
    
    Args:
        answer (str): 原始答案文本
        
    Returns:
        str: 标准化后的答案文本

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


def classify_question(question):
    """
    问题类型自动分类
    
    该函数通过关键词匹配，自动将问题分类到预定义的类型中。
    问题类型用于后续的分类评估和统计分析。
    
    问题类型分类：
    - counting（计数问题）：询问数量，如"有多少个..."
    - attribute（属性问题）：询问属性，如"什么颜色..."
    - spatial（空间关系）：询问位置关系，如"左边是什么..."
    - reading（文字识别）：询问图像中的文字内容
    - yesno（是否问题）：询问是/否，如"是否是..."
    - identification（识别问题）：询问对象名称，如"这是什么..."
    - other（其他）：未分类的问题
    
    Args:
        question (str): 问题文本
        
    Returns:
        str: 问题类型名称

    """
    # 转换为小写，便于关键词匹配
    q = question.lower()

    # 定义各类问题的关键词
    categories = {
        # 计数类问题关键词
        'counting': ['how many', '多少', 'count', 'number of', '数量', 'how much'],
        
        # 属性类问题关键词
        'attribute': ['what color', 'what brand', 'what type', 'what kind', 'what year',
                      'what time', '颜色', '品牌', '类型', '年份', '时间', 'what is the'],
        
        # 空间关系类问题关键词
        'spatial': ['where', 'what is on the left', 'what is on the right', 'what is in front',
                    '位置', '左边', '右边', '前面', '后面', '上面', '下面'],
        
        # 文字识别类问题关键词
        'reading': ['what does it say', 'what does the sign say', 'what does the text say',
                    'what is written', 'read', '读取', '文字', '写的', '说什么', 'what word'],
        
        # 是否类问题关键词
        'yesno': ['is this', 'are these', 'was the', 'does her shirt say', '是否', '是不是',
                  'does the', 'is there', 'are there'],
        
        # 识别类问题关键词
        'identification': ['who is', 'what is the name', 'what is this', 'who was',
                          '谁', '名称', '是什么', 'what does']
    }

    # 遍历各类别，检查问题是否包含该类别的关键词
    for cat, keywords in categories.items():
        if any(kw in q for kw in keywords):
            return cat
            
    # 未匹配到任何类别，返回"other"
    return 'other'


# ==============================================================================
# 第九部分：评估指标计算函数
# ==============================================================================


def compute_exact_match(pred, targets):
    """
    精确匹配评估
    
    该函数检查预测答案是否与标准答案完全匹配（标准化后）。
    精确匹配是最严格的评估方式。
    
    Args:
        pred (str): 模型预测的答案
        targets (list): 标准答案列表
        
    Returns:
        bool: 如果有任何一个标准答案与预测完全匹配，返回True

    """
    # 标准化预测答案
    pred_norm = normalize_answer(pred)
    
    for target in targets:
        if pred_norm == normalize_answer(target):
            return True
            
    return False


def compute_fuzzy_match(pred, targets):
    """
    模糊匹配评估
    
    该函数使用多种策略进行宽松的答案匹配，
    包括子串匹配和词汇重叠度计算。
    
    匹配策略（优先级从高到低）：
    1. 精确匹配：标准化后完全相同
    2. 子串匹配：标准答案是预测的子集
    3. 词汇重叠：词汇重叠度超过70%
    
    Args:
        pred (str): 模型预测的答案
        targets (list): 标准答案列表
        
    Returns:
        bool: 如果满足任一匹配条件，返回True

    """
    # 标准化预测答案
    pred_norm = normalize_answer(pred)

    for target in targets:
        target_norm = normalize_answer(target)

        # 策略1：精确匹配
        if pred_norm == target_norm:
            return True
            
        # 策略2：子串匹配（标准答案是预测的子集）
        if target_norm in pred_norm:
            return True
            
        # 策略3：词汇重叠度匹配
        # 条件：预测是标准的超集，且重叠度超过70%
        if pred_norm in target_norm and len(pred_norm) > 3:
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
    
    该函数计算预测答案的综合准确率，
    考虑多种匹配策略并返回最高匹配分数。
    
    评分机制：
    - 精确匹配：1.0分
    - 标准答案是预测的子集：0.9分
    - 预测是标准的超集：0.8分
    - 词汇重叠：按比例计算分数
    
    Args:
        pred (str): 模型预测的答案
        targets (list): 标准答案列表
        
    Returns:
        bool: 如果最高分数 >= 0.6，返回True

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


def create_category_chart(category_stats, output_dir):
    """
    创建问题类型统计图表
    
    该函数生成两个子图：
    1. 各类别的样本数量统计（堆叠柱状图）
    2. 各类别的准确率统计（彩色柱状图）
    
    Args:
        category_stats (dict): 分类统计数据
            格式：{'counting': {'correct': 10, 'total': 20}, ...}
        output_dir (str): 输出目录路径
        
    Returns:
        str: 保存的图表文件路径
        
    颜色编码：
        - 绿色：准确率 > 50%（良好）
        - 橙色：30% < 准确率 <= 50%（一般）
        - 红色：准确率 <= 30%（需改进）

    """
    # 提取数据
    categories = list(category_stats.keys())
    totals = [stats['total'] for stats in category_stats.values()]
    correct = [stats['correct'] for stats in category_stats.values()]
    
    # 计算各类别准确率
    accuracies = [c / t if t > 0 else 0 for c, t in zip(correct, totals)]

    # 创建图表（1行2列）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1：样本数量统计
    x = np.arange(len(categories))
    ax1.bar(x - 0.175, totals, 0.35, label='总计', color='steelblue')
    ax1.bar(x + 0.175, correct, 0.35, label='正确', color='green')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend()
    ax1.set_title('问题类型统计')

    # 子图2：准确率统计
    colors = ['green' if acc > 0.5 else 'orange' if acc > 0.3 else 'red' for acc in accuracies]
    ax2.bar(categories, accuracies, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.axhline(y=0.5, color='gray', linestyle='--', label='50%基线')
    ax2.set_title('各问题类型准确率')
    ax2.legend()

    # 添加准确率标签
    for i, acc in enumerate(accuracies):
        if acc > 0:
            ax2.text(i, acc + 0.02, f'{acc:.1%}', ha='center', fontsize=9)

    # 调整布局并保存
    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'category_statistics.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"分类统计图表已保存到: {chart_path}")
    return chart_path


def create_visualization(results, image_dir, output_dir, num_samples=20, show_clip_info=False):
    """
    创建VQA评估结果的可视化展示
    
    该函数生成一个网格布局的图像，展示评估样本的预测结果。
    样本包括正确预测和错误预测，便于直观分析模型表现。
    
    Args:
        results (list): 评估结果列表
        image_dir (str): 图像目录路径
        output_dir (str): 输出目录路径
        num_samples (int): 展示的样本数量，默认20
        show_clip_info (bool): 是否显示CLIP相关信息，默认False
        
    Returns:
        str: 保存的可视化图像路径

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
                f"ID:{result['id']} [{result['category']}] {status}",
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
    
    该函数为PIL/Pillow图像库提供中文字体支持。
    尝试多种候选字体，返回第一个可用的字体。
    
    Args:
        size (int): 字体大小
        
    Returns:
        ImageFont: PIL字体对象，如果都不可用则返回默认字体

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
    
    该函数生成一个并排对比图：
    - 左侧：原始图像
    - 右侧：分类/检索结果（概率条形图）
    
    Args:
        image_path (str): 图像文件路径
        labels (list): 标签/类别列表
        probs (list): 对应概率列表
        output_path (str): 输出文件路径
        task_type (str): 任务类型，"classification"或"retrieval"
        
    Returns:
        str: 保存的文件路径
        
    使用示例：
        labels = ["猫", "狗", "车"]
        probs = [0.8, 0.15, 0.05]
        create_visualization_comparison("image.jpg", labels, probs, "output.png")
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
    
    该函数生成一个横向拼接图：
    - 左侧：查询文本说明
    - 右侧：检索结果图像，按相似度排序
    
    Args:
        query_text (str): 查询文本
        results (list): 检索结果列表
            格式：[{"image": "path", "similarity": 0.9, "rank": 1}, ...]
        image_folder (str): 图像文件夹路径
        output_path (str): 输出文件路径
        
    Returns:
        str: 保存的文件路径

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
    
    该函数生成一个并排对比图，用于展示VQA任务的结果。
    
    Args:
        image_path (str): 图像文件路径
        question (str): 问题文本
        answer (str): 模型答案
        output_path (str): 输出文件路径
        
    Returns:
        str: 保存的文件路径

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


def compute_metrics_base(results, category_stats):
    """
    计算基础评估指标
    
    该函数根据评估结果计算整体和各分类别的准确率。
    
    Args:
        results (list): 评估结果列表
        category_stats (dict): 分类统计数据
        
    Returns:
        dict: 包含各项指标的字典
            - overall_accuracy: 总体准确率
            - total_samples: 总样本数
            - correct_samples: 正确样本数
            - category_accuracy: 各分类准确率
            - category_stats: 分类统计详情
    """
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


# ==============================================================================
# 第十二部分：结果保存函数
# ==============================================================================


def save_classification_result(image_path, labels, all_probs, output_dir):
    """
    保存分类结果到JSON文件
    
    Args:
        image_path (str): 图像路径
        labels (list): 类别标签列表
        all_probs (list): 对应概率列表
        output_dir (str): 输出目录
        
    Returns:
        str: 保存的文件路径
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
    
    Args:
        query_text (str): 查询文本
        results (list): 检索结果列表
        output_dir (str): 输出目录
        
    Returns:
        str: 保存的文件路径
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
    
    Args:
        image_path (str): 图像路径
        question (str): 问题文本
        answer (str): 模型答案
        output_dir (str): 输出目录
        
    Returns:
        str: 保存的文件路径
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