"""
VQA 公共功能模块
================

本文件包含VQA评估系统的公共功能模块，用于被多个评估脚本共享。
主要功能包括：

1. 模型初始化和客户端配置
2. 数据加载和预处理
3. 文本处理和标准化
4. 问题分类系统
5. 图像处理工具
6. VQA推理引擎
7. 评估指标计算
8. 可视化和报告生成

设计原则：
- 模块化：每个函数负责单一功能
- 可复用：支持在多个脚本中共享使用
- 可扩展：便于后续功能扩展和修改
- 兼容性：支持不同评估系统的需求
"""

# 导入所需的标准库和第三方库
import json           # 用于处理JSON数据
import os            # 用于文件和目录操作
import re            # 用于正则表达式处理
from PIL import Image  # 用于图像处理
import matplotlib.pyplot as plt  # 用于数据可视化
import matplotlib.font_manager as fm  # 用于字体管理
import numpy as np    # 用于数值计算
from datetime import datetime  # 用于处理日期和时间
from openai import OpenAI  # 用于调用OpenAI兼容的API
from io import BytesIO  # 用于处理字节流
import base64  # 用于base64编码
import time  # 用于控制时间间隔

# 设置matplotlib后端为非交互式模式，适合在服务器环境中使用
plt.switch_backend('Agg')

# ====================
# 中文字体配置
# ====================

# 获取系统中可用的所有字体名称列表
available_fonts = [f.name for f in fm.fontManager.ttflist]
# 定义常用的中文字体列表，按优先级排序
chinese_fonts = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Droid Sans Fallback']
# 初始化选中的字体变量
selected_font = None

# 遍历中文字体列表，选择系统中可用的第一个字体
for font in chinese_fonts:
    if font in available_fonts:
        selected_font = font  # 找到可用字体，赋值给selected_font
        break  # 找到后立即退出循环

# 根据是否找到合适的中文字体进行配置
if selected_font:
    # 如果找到合适的中文字体，设置为matplotlib的默认字体
    plt.rcParams['font.family'] = selected_font
    print(f"使用中文字体: {selected_font}")  # 打印使用的字体名称
else:
    # 如果没有找到合适的中文字体，尝试使用系统默认中文字体
    print("未找到合适的中文字体，尝试使用系统默认中文字体")
    # 设置字体列表，包含多种备选字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    # 设置解决负号显示问题，确保负号能正确显示
    plt.rcParams['axes.unicode_minus'] = False


# =====================
# 1. 模型和客户端初始化
# =====================

def load_model(api_key):
    """
    初始化Qwen3-VL API客户端
    
    Args:
        api_key (str): DashScope API密钥
    
    Returns:
        tuple: (client, model_name) - API客户端和模型名称
    """
    # 打印初始化信息
    print("初始化 Qwen3-VL API 客户端")
    # 创建OpenAI兼容的客户端实例，配置连接到阿里云DashScope服务
    client = OpenAI(
        api_key=api_key,  # 使用提供的API密钥进行身份验证
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # DashScope API的兼容模式端点
    )
    # 指定要使用的Qwen3-VL模型版本
    model_name = "qwen3-vl-8b-instruct"
    # 打印使用的模型名称
    print(f"使用模型: {model_name}")
    # 返回客户端实例和模型名称
    return client, model_name


# =====================
# 2. 数据加载和预处理
# =====================

def load_metadata(data_dir):
    """
    加载数据集元数据
    
    Args:
        data_dir (str): 数据目录路径
    
    Returns:
        list: 包含图像、问题和答案的元数据列表
    """
    # 构建metadata.json文件的完整路径
    metadata_path = os.path.join(data_dir, "metadata.json")
    # 打开并读取metadata.json文件，使用utf-8编码
    with open(metadata_path, 'r', encoding='utf-8') as f:
        # 将JSON数据解析为Python对象
        metadata = json.load(f)
    # 打印加载的元数据数量
    print(f"加载了 {len(metadata)} 条元数据")
    # 返回加载的元数据列表
    return metadata


# =====================
# 3. 文本处理和标准化
# =====================

def normalize_answer(answer):
    """
    答案标准化处理
    
    对模型生成的答案进行清理和标准化，便于后续的匹配比较。
    处理包括：转小写、去除标点、标准化空格等。
    
    Args:
        answer (str): 原始答案文本
    
    Returns:
        str: 标准化后的答案文本
    """
    # 如果答案为空，直接返回空字符串
    if not answer:
        return ""
    # 将答案转换为小写并去除首尾空格
    answer = answer.lower().strip()
    # 使用正则表达式去除所有标点符号，用空格替代
    answer = re.sub(r'[^\w\s]', ' ', answer)
    # 使用正则表达式将多个连续空格替换为单个空格
    answer = re.sub(r'\s+', ' ', answer)
    # 去除处理后可能产生的首尾空格
    return answer.strip()


def classify_question(question):
    """
    问题类型自动分类
    
    基于关键词匹配对VQA问题进行分类，
    支持计数、属性、空间关系、文字识别等类型。
    
    Args:
        question (str): 问题文本
    
    Returns:
        str: 问题类型标签
    """
    # 将问题转换为小写，方便后续关键词匹配
    q = question.lower()
    
    # 定义各类问题的关键词字典，键为问题类型，值为关键词列表
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
        # 是非类问题关键词
        'yesno': ['is this', 'are these', 'was the', 'does her shirt say', '是否', '是不是',
                  'does the', 'is there', 'are there'],
        # 识别类问题关键词
        'identification': ['who is', 'what is the name', 'what is this', 'who was',
                          '谁', '名称', '是什么', 'what does']
    }
    
    # 遍历所有问题类型和对应的关键词列表
    for cat, keywords in categories.items():
        # 检查问题中是否包含当前类型的任何一个关键词
        if any(kw in q for kw in keywords):
            return cat  # 如果包含，返回当前问题类型
    # 如果没有匹配到任何类型，返回'other'表示其他类型
    return 'other'


# =====================
# 4. 图像处理工具
# =====================

def image_to_base64(image_path):
    """
    将图像转换为base64编码
    
    用于API调用时的图像传输。
    包含图像预处理：格式转换、尺寸调整、质量优化等。
    
    Args:
        image_path (str): 图像文件路径
    
    Returns:
        str: base64编码的图像字符串，失败返回None
    """
    try:
        # 打开图像文件
        with Image.open(image_path) as img:
            # 将图像转换为RGB格式，确保与API兼容
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 设置图像最大尺寸限制为1024像素
            max_size = 1024
            # 检查图像是否超过最大尺寸
            if max(img.size) > max_size:
                # 计算缩放比例
                ratio = max_size / max(img.size)
                # 计算新的图像尺寸
                new_size = tuple(int(dim * ratio) for dim in img.size)
                # 使用LANCZOS插值算法调整图像大小，这是一种高质量的图像缩放算法
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # 创建BytesIO对象，用于临时存储图像数据
            buffered = BytesIO()
            # 将图像保存为JPEG格式到BytesIO对象，质量设置为85%（平衡大小和清晰度）
            img.save(buffered, format="JPEG", quality=85)
            # 将BytesIO对象中的数据转换为base64编码的字符串
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            # 返回base64编码的图像字符串
            return img_str
    except Exception as e:
        # 捕获并打印图像处理过程中可能出现的错误
        print(f"图片处理错误 {image_path}: {e}")
        # 处理失败时返回None
        return None


# =====================
# 5. VQA推理引擎
# =====================

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
    # 尝试多次调用API，处理可能的临时错误
    for attempt in range(max_retries):
        try:
            # 对图像进行预处理并转换为base64编码
            base64_image = image_to_base64(image_path)
            # 检查图像处理是否成功
            if not base64_image:
                return f"错误: 无法处理图片"
            
            # 构建API调用的消息格式
            messages = [
                {
                    "role": "user",  # 消息角色为用户
                    "content": [
                        # 图像URL部分，包含base64编码的图像数据
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"  # 数据URL格式
                            }
                        },
                        # 文本问题部分
                        {
                            "type": "text",
                            # 问题文本，添加提示要求简短回答
                            "text": question + "\n请简短回答，只回答关键信息，不需要解释。"
                        }
                    ]
                }
            ]
            
            # 调用API进行视觉问答推理
            response = client.chat.completions.create(
                model=model_name,  # 指定使用的模型
                messages=messages,  # 传递构建好的消息
                max_tokens=128,    # 限制回答长度，防止回答过长
                temperature=0.1,   # 设置低温度，确保回答更加稳定和确定性
                stream=False       # 禁用流式输出，获取完整回答后再返回
            )
            
            # 从API响应中提取模型回答
            answer = response.choices[0].message.content.strip()
            # 返回模型回答
            return answer
            
        except Exception as e:
            # 捕获API调用过程中可能出现的任何异常
            print(f"API调用错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            # 检查是否还有重试机会
            if attempt < max_retries - 1:
                time.sleep(2)  # 等待2秒后重试
            else:
                # 所有重试都失败，返回错误信息
                return f"错误: {str(e)}"


# =====================
# 6. 评估指标计算
# =====================

def compute_exact_match(pred, targets):
    """
    精确匹配评估
    
    检查预测答案是否与任何一个目标答案完全匹配（经过标准化处理后）。
    
    Args:
        pred (str): 模型预测的答案
        targets (list): 参考答案列表，通常包含多个标注答案
    
    Returns:
        bool: 如果预测答案与任何一个参考答案精确匹配，返回True；否则返回False
    """
    # 对预测答案进行标准化处理
    pred_norm = normalize_answer(pred)
    # 遍历所有参考答案
    for target in targets:
        # 对当前参考答案进行标准化处理
        target_norm = normalize_answer(target)
        # 检查标准化后的预测答案是否与标准化后的参考答案完全匹配
        if pred_norm == target_norm:
            return True  # 找到匹配，返回True
    # 遍历完所有参考答案都没有找到匹配，返回False
    return False


def compute_fuzzy_match(pred, targets):
    """
    模糊匹配算法
    
    实现多种模糊匹配策略，提高评估的鲁棒性。
    包括精确匹配、子串匹配、单词重叠度计算等。
    
    Args:
        pred (str): 模型预测的答案
        targets (list): 参考答案列表
    
    Returns:
        bool: 如果预测答案与任何一个参考答案模糊匹配，返回True；否则返回False
    """
    # 对预测答案进行标准化处理
    pred_norm = normalize_answer(pred)
    
    # 遍历所有参考答案
    for target in targets:
        # 对当前参考答案进行标准化处理
        target_norm = normalize_answer(target)
        
        # 1. 精确匹配检查
        if pred_norm == target_norm:
            return True  # 精确匹配，直接返回True
        
        # 2. 子串匹配检查：参考答案是否是预测答案的子串
        if target_norm in pred_norm:
            return True  # 参考答案是预测答案的子串，返回True
        
        # 3. 包含关系检查：预测答案是否是参考答案的子串，且长度大于3
        if pred_norm in target_norm and len(pred_norm) > 3:
            return True  # 预测答案是参考答案的子串且长度足够，返回True
        
        # 4. 单词重叠度检查
        # 将预测答案拆分为单词集合
        pred_words = set(pred_norm.split())
        # 将参考答案拆分为单词集合
        target_words = set(target_norm.split())
        # 检查参考答案单词集合是否非空，避免除以零错误
        if target_words:
            # 计算单词重叠率：共同单词数 / 参考答案单词数
            overlap_ratio = len(pred_words & target_words) / len(target_words)
            # 如果重叠率超过70%，认为匹配成功
            if overlap_ratio > 0.7:
                return True
    
    # 所有匹配策略都失败，返回False
    return False


def compute_accuracy(pred, targets):
    """
    计算预测答案的准确率
    
    使用多种匹配策略综合评估预测答案的质量。
    支持精确匹配、模糊匹配和部分匹配。
    
    Args:
        pred (str): 模型预测的答案
        targets (list): 参考答案列表（通常有多个标注答案）
    
    Returns:
        bool: 预测是否正确
    """
    # 对预测答案进行标准化处理
    pred_norm = normalize_answer(pred)
    
    # 初始化最佳匹配和最佳分数
    best_match = None
    best_score = 0
    
    # 遍历所有目标答案，找出最佳匹配
    for target in targets:
        # 对当前目标答案进行标准化处理
        target_norm = normalize_answer(target)
        
        # 1. 精确匹配检查
        if pred_norm == target_norm:
            return True  # 精确匹配，直接返回True
        
        # 初始化当前匹配分数
        score = 0
        
        # 2. 部分匹配评分
        if target_norm in pred_norm:
            score = 0.9  # 高分：参考答案完全包含在预测答案中
        elif pred_norm in target_norm:
            score = 0.8  # 中高分：预测答案包含在参考答案中
        else:
            # 3. 单词重叠度评分
            # 计算共同单词集合
            common = set(pred_norm.split()) & set(target_norm.split())
            # 检查参考答案是否有单词，避免除以零错误
            if target_norm.split():
                # 计算重叠率：共同单词数 / 参考答案单词数
                score = len(common) / len(set(target_norm.split()))
        
        # 更新最佳匹配和最佳分数
        if score > best_score:
            best_score = score  # 更新最佳分数
            best_match = target  # 更新最佳匹配的参考答案
    
    # 设置评分阈值，高于60%认为匹配成功
    return best_score >= 0.6


# =====================
# 7. 可视化和报告生成
# =====================

def create_category_chart(category_stats, output_dir):
    """
    创建问题类型统计图表
    
    生成柱状图展示各问题类型的样本数量分布和准确率表现。
    包含两个子图：样本数量统计和各类型准确率。
    
    Args:
        category_stats (dict): 分类统计数据，键为问题类型，值为包含'correct'和'total'的字典
        output_dir (str): 输出目录路径
    
    Returns:
        str: 图表保存路径
    """
    # 提取问题类型列表
    categories = list(category_stats.keys())
    # 提取各类型的总样本数
    totals = [stats['total'] for stats in category_stats.values()]
    # 提取各类型的正确样本数
    correct = [stats['correct'] for stats in category_stats.values()]
    # 计算各类型的准确率，避免除以零错误
    accuracies = [c / t if t > 0 else 0 for c, t in zip(correct, totals)]
    
    # 创建1行2列的子图布局，设置图大小为14x5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1：样本数量统计
    x = np.arange(len(categories))  # 创建x轴坐标
    # 绘制总样本数柱状图，位置向左偏移0.175
    ax1.bar(x - 0.175, totals, 0.35, label='总计', color='steelblue')
    # 绘制正确样本数柱状图，位置向右偏移0.175
    ax1.bar(x + 0.175, correct, 0.35, label='正确', color='green')
    # 设置x轴刻度位置
    ax1.set_xticks(x)
    # 设置x轴刻度标签为问题类型，旋转45度，右对齐
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    # 添加图例
    ax1.legend()
    # 设置子图标题
    ax1.set_title('问题类型统计')
    
    # 子图2：各类型准确率
    # 根据准确率设置柱状图颜色：绿色>50%，橙色>30%，红色<=30%
    colors = ['green' if acc > 0.5 else 'orange' if acc > 0.3 else 'red' for acc in accuracies]
    # 绘制准确率柱状图
    ax2.bar(categories, accuracies, color=colors)
    # 设置x轴刻度位置
    ax2.set_xticks(x)
    # 设置x轴刻度标签为问题类型，旋转45度，右对齐
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    # 添加50%准确率基线
    ax2.axhline(y=0.5, color='gray', linestyle='--', label='50%基线')
    # 设置子图标题
    ax2.set_title('各问题类型准确率')
    # 添加图例
    ax2.legend()
    
    # 在柱状图上标注百分比
    for i, acc in enumerate(accuracies):
        if acc > 0:  # 只标注准确率大于0的情况
            # 在柱状图上方添加百分比文本，居中对齐
            ax2.text(i, acc + 0.02, f'{acc:.1%}', ha='center', fontsize=9)
    
    # 调整布局，避免元素重叠
    plt.tight_layout()
    # 构建图表保存路径
    chart_path = os.path.join(output_dir, 'category_statistics.png')
    # 保存图表，设置DPI为150，确保清晰度
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    # 关闭图表，释放内存
    plt.close()
    # 打印保存路径
    print(f"分类统计图表已保存到: {chart_path}")
    # 返回图表保存路径
    return chart_path


def create_visualization(results, image_dir, output_dir, num_samples=20, show_clip_info=False):
    """
    创建VQA评估结果的可视化展示
    
    生成包含成功和失败案例的图像网格，直观展示模型的评估表现。
    每个子图显示原始图像、问题、模型答案和评估结果。
    
    Args:
        results (list): 评估结果列表，包含每个样本的评估详情
        image_dir (str): 图像目录路径
        output_dir (str): 输出目录路径
        num_samples (int): 要展示的样本数量，默认20个
        show_clip_info (bool): 是否显示CLIP相关信息，默认不显示
    
    Returns:
        str: 可视化图像的保存路径
    """
    # 分离成功和失败的案例
    success = [r for r in results if r['is_correct']]  # 筛选成功案例
    failure = [r for r in results if not r['is_correct']]  # 筛选失败案例
    
    # 初始化选中的样本列表
    selected = []
    # 平衡选择成功和失败的案例，各占一半
    num_success = min(num_samples // 2, len(success))  # 成功案例数量
    num_failure = min(num_samples - num_success, len(failure))  # 失败案例数量
    
    # 随机选择成功案例
    if num_success > 0:
        # 随机生成指定数量的索引，不重复
        indices = np.random.choice(len(success), num_success, replace=False)
        # 根据索引选择成功案例
        for idx in indices:
            selected.append(success[idx])
    
    # 随机选择失败案例
    if num_failure > 0:
        # 随机生成指定数量的索引，不重复
        indices = np.random.choice(len(failure), num_failure, replace=False)
        # 根据索引选择失败案例
        for idx in indices:
            selected.append(failure[idx])
    
    # 随机打乱选中样本的顺序
    np.random.shuffle(selected)
    
    # 计算子图布局：4列，行数根据样本数量调整
    cols, rows = 4, (len(selected) + 3) // 4
    # 创建子图网格
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    # 将二维坐标轴数组展平为一维，方便遍历
    axes = axes.flatten()
    
    # 为每个选中的案例创建子图
    for idx, result in enumerate(selected):
        # 检查是否超出坐标轴数量
        if idx >= len(axes):
            break
        # 获取当前坐标轴
        ax = axes[idx]
        try:
            # 加载并显示图像
            img = Image.open(os.path.join(image_dir, result['image_file']))
            ax.imshow(img)  # 显示图像
            ax.axis('off')  # 隐藏坐标轴
            
            # 设置结果状态文本和颜色
            status = "[成功]" if result['is_correct'] else "[失败]"
            color = 'green' if result['is_correct'] else 'red'
            
            # 构建标题信息列表
            title_parts = [
                f"ID:{result['id']} [{result['category']}] {status}",  # 样本ID、类型和状态
                f"问题: {result['question'][:35]}...",  # 问题文本，截断显示
            ]
            
            # 根据是否显示CLIP信息调整答案显示
            if show_clip_info and 'clip_score' in result:
                # 选择要显示的答案字段：优先显示final_answer，否则显示model_answer
                answer_field = 'final_answer' if 'final_answer' in result else 'model_answer'
                # 构建CLIP得分信息
                clip_info = f"\nCLIP得分: {result['clip_score']:.2f}" if result['clip_score'] > 0 else ""
                # 构建CLIP重排序信息
                rerank_info = " (CLIP优化)" if result.get('clip_reranked', False) else ""
                
                # 添加预测答案信息
                title_parts.append(
                    f"预测: {result[answer_field][:25]}{rerank_info}{clip_info}"
                )
                # 如果有真实答案，添加真实答案信息
                if 'ground_truth' in result:
                    title_parts.append(
                        f"真实: {result['ground_truth'][:25]}"
                    )
            else:
                # 不显示CLIP信息时的答案字段选择
                answer_field = 'model_answer' if 'model_answer' in result else 'final_answer'
                # 添加答案信息
                title_parts.append(
                    f"答案: {result[answer_field][:30]}"
                )
            
            # 设置子图标题，使用换行符连接各部分信息
            ax.set_title(
                "\n".join(title_parts),
                fontsize=8, color=color  # 设置字体大小和颜色
            )
        except Exception as e:
            # 图像加载失败的处理
            ax.text(0.5, 0.5, f"图像加载失败", ha='center', va='center')  # 显示错误信息
            ax.axis('off')  # 隐藏坐标轴
    
    # 隐藏多余的子图
    for ax in axes.flat[len(selected):]:
        ax.axis('off')
    
    # 调整布局，避免元素重叠
    plt.tight_layout()
    # 构建可视化结果保存路径
    viz_path = os.path.join(output_dir, 'vqa_visualization.png')
    # 保存可视化结果，设置DPI为150
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    # 关闭图表，释放内存
    plt.close()
    # 打印保存路径
    print(f"可视化结果已保存到: {viz_path}")
    # 返回保存路径
    return viz_path


def compute_metrics_base(results, category_stats):
    """
    计算基础评估指标
    
    汇总评估结果，计算总体和各分类的准确率等指标。
    
    Args:
        results (list): 评估结果列表
        category_stats (dict): 分类统计字典
    
    Returns:
        dict: 包含各种评估指标的字典
    """
    # 计算总样本数
    total = len(results)
    # 计算正确样本数
    correct = sum(1 for r in results if r['is_correct'])
    # 计算总体准确率，避免除以零错误
    overall_accuracy = correct / total if total > 0 else 0
    
    # 初始化各问题类型的准确率字典
    category_accuracy = {}
    # 遍历各问题类型的统计数据
    for cat, stats in category_stats.items():
        # 计算当前类型的准确率，避免除以零错误
        category_accuracy[cat] = stats['correct'] / stats['total'] if stats['total'] > 0 else None
    
    # 返回包含所有评估指标的字典
    return {
        'overall_accuracy': overall_accuracy,  # 总体准确率
        'total_samples': total,  # 总样本数
        'correct_samples': correct,  # 正确样本数
        'category_accuracy': category_accuracy,  # 各类型准确率
        'category_stats': category_stats  # 原始分类统计数据
    }
