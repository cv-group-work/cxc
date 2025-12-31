# Qwen3-VL + CLIP 视觉问答评估系统

## 项目概述

本项目是一个基于多模态模型的视觉问答(VQA)评估系统，结合了阿里云通义千问Qwen3-VL模型的强大VQA能力和OpenAI CLIP模型的图像-文本匹配精度，实现了高精度的图像问答能力评估。

### 核心亮点

- **多模态融合**：结合Qwen3-VL的VQA推理能力与CLIP的图像-文本相似度计算
- **智能重排序**：通过CLIP相似度优化Qwen3-VL的答案质量
- **消融实验**：完整对比有无CLIP重排序的效果差异
- **多维度评估**：支持精确匹配、模糊匹配、问题分类等多种评估指标
- **模块化设计**：公共功能提取到独立模块，便于维护和扩展
- **中文友好**：完整的中文注释、报告和可视化结果
- **详细可视化**：多样化的评估结果可视化和分析报告

## 技术架构

### 核心模型

| 模型 | 类型 | 功能 |
|------|------|------|
| Qwen3-VL-8B-Instruct | 视觉语言模型 | 强大的多模态理解和图像问答能力 |
| CLIP ViT-B/32 | 对比学习模型 | 优秀的图像-文本相似度计算和零样本分类能力 |

### 系统流程

```
输入图像 + 问题 → Qwen3-VL API推理 → 初始答案 → CLIP相似度验证 → 最终优化答案 → 评估结果
```

## 项目结构

```
VQA/
├── README.md                          # 项目说明文档
├── config.py                          # 项目配置文件
├── .env                               # 环境变量配置（包含API密钥等敏感信息）
├── src/                               # 源代码目录
│   ├── 0_clip_demo.py                 # CLIP模型功能演示脚本
│   ├── 0_qwen_reproduce.py            # Qwen3-VL模型基础推理演示
│   ├── 1_vqa_evaluation.py            # 基于Qwen3-VL API的单一模型评估
│   ├── 1_vqa_evaluation_with_clip.py  # Qwen3-VL + CLIP多模型融合评估（含消融实验）
│   └── vqa_common.py                  # 核心公共功能模块
├── data/                              # 数据目录
│   ├── images/                        # 测试图像数据集
│   │   ├── metadata.json              # 数据集元数据（包含图像、问题、答案）
│   │   └── *.jpg                      # 测试图像文件
│   └── results/                       # 评估结果输出目录
│       ├── clip_demo/                 # CLIP演示结果
│       ├── qwen_demo/                 # Qwen3-VL演示结果
│       ├── vqa_results_api/           # 单一模型API评估结果
│       └── vqa_results_ablation/      # 多模型消融实验结果
└── doc/                               # 项目文档目录
    ├── 1_vqa_evaluation.md            # 单一模型评估详细文档
    ├── 1_vqa_evaluation_with_clip.md  # 多模型融合评估详细文档
    └── vqa_common.md                  # 公共功能模块文档
```

### 目录结构说明

| 目录 | 功能描述 |
|------|----------|
| `src/` | 包含所有核心源代码文件，按功能模块化设计 |
| `data/` | 包含测试数据和评估结果，便于数据管理和结果归档 |
| `doc/` | 包含详细的模块文档，便于开发者理解和扩展 |

### 核心文件说明

| 文件名 | 功能描述 |
|--------|----------|
| `vqa_common.py` | 核心公共功能模块，包含模型加载、数据处理、评估指标计算、可视化等关键功能 |
| `1_vqa_evaluation.py` | 基于Qwen3-VL API的基础VQA评估脚本，不使用CLIP重排序，生成基础评估报告 |
| `1_vqa_evaluation_with_clip.py` | 结合CLIP的增强版VQA评估脚本，包含完整的消融实验功能，对比有无CLIP重排序的效果 |
| `0_clip_demo.py` | CLIP模型功能演示脚本，展示零样本分类、图像检索和相似度计算等功能 |
| `0_qwen_reproduce.py` | Qwen3-VL模型基础演示脚本，展示单张图像问答、模型加载和推理等基本功能 |
| `config.py` | 项目配置文件，包含模型路径、数据目录、评估参数等配置项 |
| `.env` | 环境变量配置文件，包含API密钥等敏感信息，不建议提交到版本控制 |
| `metadata.json` | 数据集元数据文件，包含图像文件名、问题、标准答案等信息，用于评估 |

### 结果文件说明

| 目录 | 功能描述 |
|------|----------|
| `clip_demo/` | CLIP模型演示结果，包含分类和检索结果的可视化和JSON数据 |
| `qwen_demo/` | Qwen3-VL模型演示结果，包含单张图像问答的结果和可视化 |
| `vqa_results_api/` | 单一模型API评估结果，包含评估报告、统计图表和详细结果 |
| `vqa_results_ablation/` | 多模型消融实验结果，包含有无CLIP重排序的对比分析和可视化

## 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 环境变量配置

在项目根目录创建 `.env` 文件，添加以下内容：

```bash
# 阿里云DashScope API密钥
DASHSCOPE_API_KEY=your_api_key_here

# CUDA设备配置（可选）
CUDA_VISIBLE_DEVICES=0
```

### 3. 数据准备

将测试图像放置在 `data/images/` 目录中，并确保存在 `metadata.json` 文件，格式如下：

```json
[
    {
        "id": "image_001",
        "image_file": "0.jpg",
        "question": "这张图片中有什么？",
        "answers": ["一只猫", "猫", "猫咪"]
    }
]
```

## 使用指南

### 基础演示

#### CLIP模型演示

```bash
python src/0_clip_demo.py
```

功能：
- 零样本图像分类
- 文本检索图像
- 图像-文本相似度计算

#### Qwen3-VL模型演示

```bash
python src/0_qwen_reproduce.py
```

功能：
- 单张图像问答
- 模型加载和推理
- 基础多模态理解

### 评估系统

#### 单一模型评估

```bash
python src/1_vqa_evaluation.py
```

功能：
- Qwen3-VL API版本的VQA性能评估
- 问题类型分类评估
- 多种匹配策略（精确匹配、模糊匹配）
- 可视化结果和统计报告

#### 多模型融合评估（推荐）

```bash
python src/1_vqa_evaluation_with_clip.py
```

增强功能：
- Qwen3-VL + CLIP组合评估
- CLIP重排序机制优化答案质量
- 完整的消融实验对比
- 详细的对比分析报告

## 评估指标

### 主要指标

- **总体准确率**: 正确预测样本数 / 总样本数
- **分类准确率**: 按问题类型分别计算准确率
- **模糊匹配率**: 使用模糊匹配策略的准确率
- **CLIP优化率**: 通过CLIP重排序改善的样本比例

### 问题类型分类

- **计数问题** (counting): "有多少个..."、"How many..."
- **属性问题** (attribute): "什么颜色"、"什么品牌"
- **空间关系** (spatial): "左边是什么"、"位置关系"
- **文字识别** (reading): "写了什么"、"读取文字"
- **是否问题** (yesno): "是否是..."、"有没有..."
- **识别问题** (identification): "这是什么"、"谁在..."

### 匹配策略

1. **精确匹配**: 标准化后完全相同的答案
2. **子串匹配**: 一个答案包含另一个
3. **单词重叠**: 词汇重叠度超过70%
4. **CLIP相似度**: 基于视觉相似度的匹配

## 输出文件

### 基础评估结果

```
data/results/vqa_results_api/
├── all_results.json              # 完整评估结果
├── evaluation_report.json        # 评估报告
├── evaluation_summary.txt        # 文本摘要
├── vqa_visualization.png         # 结果可视化
└── category_statistics.png       # 分类统计图表
```

### 消融实验结果

```
data/results/vqa_results_ablation/
├── ablation_comparison.png          # 对比图表
├── ablation_report.json             # 消融实验报告
├── all_results_clip_disabled.json   # 禁用CLIP重排序的结果
├── all_results_clip_enabled.json    # 启用CLIP重排序的结果
├── category_statistics.png          # 分类统计
├── evaluation_report_clip_disabled.json  # 禁用CLIP的评估报告
├── evaluation_report_clip_enabled.json   # 启用CLIP的评估报告
├── evaluation_summary_clip_disabled.txt  # 禁用CLIP的文本摘要
├── evaluation_summary_clip_enabled.txt   # 启用CLIP的文本摘要
└── vqa_visualization.png           # 结果可视化
```

## 公共功能模块

`vqa_common.py` 模块包含以下核心功能：

| 功能类别 | 函数名 | 功能描述 |
|----------|--------|----------|
| 模型初始化 | `load_model` | 初始化Qwen3-VL API客户端 |
| 数据处理 | `load_metadata` | 加载和解析数据集元数据 |
| 数据处理 | `normalize_answer` | 答案标准化和预处理 |
| 问题分类 | `classify_question` | 自动分类问题类型 |
| VQA推理 | `vqa_inference` | 调用Qwen3-VL API进行推理 |
| 评估指标 | `compute_exact_match` | 计算精确匹配准确率 |
| 评估指标 | `compute_fuzzy_match` | 计算模糊匹配准确率 |
| 评估指标 | `compute_accuracy` | 综合评估指标计算 |
| 可视化 | `create_visualization` | 生成结果可视化 |
| 可视化 | `create_category_chart` | 生成分类统计图表 |

## 配置说明

`config.py` 文件包含以下主要配置项：

```python
# 模型配置
MODELS = {
    "clip": "openai/clip-vit-base-patch32",  # CLIP模型路径
    "qwen": "Qwen/Qwen3-VL-4B-Instruct"       # Qwen3-VL模型路径
}

# 数据目录配置
DATA_IMAGES = os.path.join("data", "images")
DATA_RESULTS = os.path.join("data", "results")

# 评估参数
SAMPLE_SIZE = 100          # 评估样本数量
CLIP_RERANK = True         # 是否启用CLIP重排序
CLIP_THRESHOLD = 0.3       # CLIP重排序阈值
```

## 扩展开发

### 添加新模型

1. 在 `config.py` 中添加模型配置
2. 在 `vqa_common.py` 中实现模型加载和推理函数
3. 更新评估脚本，集成新模型

### 添加新评估指标

1. 在 `vqa_common.py` 中添加新的评估函数
2. 更新 `compute_metrics_base` 函数，集成新指标
3. 修改可视化函数，添加新指标的展示

### 自定义数据集格式

1. 修改 `load_metadata` 函数，支持新的数据集格式
2. 调整评估逻辑，适配新的数据结构

## 常见问题

### Q: API调用失败怎么办？
A: 检查网络连接和API密钥配置，确保DashScope服务可用。系统已添加重试机制和延迟控制，可减少API调用失败的概率。

### Q: CLIP模型加载慢怎么办？
A: 首次运行会下载模型权重，后续运行会使用缓存。可以使用更小的CLIP模型版本，或预下载模型权重。

### Q: 评估结果不理想怎么办？
A: 调整 `CLIP_THRESHOLD` 参数，优化提示词模板，或尝试不同的匹配策略。

### Q: 如何增加新的问题类型？
A: 在 `classify_question` 函数中添加新的关键词分类，更新统计逻辑。

## 更新日志

### v1.0.0 (当前版本)
- ✅ 完整的Qwen3-VL + CLIP融合评估系统
- ✅ 模块化架构，公共功能提取到独立模块
- ✅ 消融实验和对比分析
- ✅ 中文界面和详细文档
- ✅ 多种评估指标和可视化
- ✅ 修复中文显示问题和TypeError错误
- ✅ 详细的行注释和代码文档

### 计划功能
- 🔄 支持更多VQA数据集
- 🔄 增加更多多模态模型
- 🔄 Web界面和API服务
- 🔄 实时评估和监控
- 🔄 模型微调和优化

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 致谢

- 阿里云通义千问团队提供Qwen3-VL模型
- OpenAI团队开源CLIP模型
- Hugging Face提供Transformers库

---

**注意**: 本项目仅供学习和研究使用，商业使用请遵循相关模型的使用协议。