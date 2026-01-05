# VQA 评估系统 - CLIP & Qwen3-VL 复现项目

## 项目概述

本项目实现了基于 CLIP 和 Qwen3-VL 的视觉问答（VQA）评估系统，包含零样本图像分类、图文检索和多模型对比评估功能。系统采用模块化设计，支持 LangChain API 调用和本地模型运行两种方式。

### 主要特性
- **CLIP 模型**：零样本图像分类和图文检索功能
- **Qwen3-VL 模型**：视觉问答推理能力
- **多模型对比**：支持消融实验和性能对比
- **可视化展示**：自动生成评估结果图表
- **模块化设计**：易于扩展和复用

## 环境准备

### 配置环境变量
创建 `.env` 文件或在环境变量中设置：
```
# 对于Qwen3-VL API访问
export DASHSCOPE_API_KEY="your_api_key_here"
```

## 运行命令

### 1. CLIP 模型基础演示
```bash
# 运行CLIP零样本分类和图文检索演示
python src/0_clip_demo.py
```

**预期输出：**
- 零样本分类结果（图像0.jpg的分类概率）
- 图文检索结果（查询"a person"的相似度排序）
- 结果保存至 `results/clip_demo/` 目录

### 2. Qwen3-VL 模型复现演示
```bash
# 运行Qwen3-VL视觉问答演示
python src/0_qwen_reproduce.py
```

**预期输出：**
- 图像5.jpg的视觉问答结果
- 结果保存至 `results/qwen_demo/` 目录

## 评测命令

### 1. 基础VQA评估（仅Qwen3-VL）
```bash
# 运行VQA评估系统
python src/1_vqa_evaluation.py
```

**评估内容：**
- 加载100个测试样本
- 对每个样本进行VQA推理
- 计算准确率和各类型问题表现
- 生成可视化图表和评估报告

**输出目录：** `results/vqa_results_api/`

### 2. 消融实验评估（Qwen3-VL + CLIP）
```bash
# 运行结合CLIP的VQA评估系统
python src/1_vqa_evaluation_with_clip.py
```

**评估内容：**
- **实验1**：基线模型（仅Qwen3-VL）
- **实验2**：Qwen3-VL + CLIP重排序
- 对比两种配置的准确率
- 分析CLIP对答案质量的改进效果

**输出目录：** `results/vqa_results_ablation/`

### 3. 完整评估流程
```bash
# 顺序运行所有评估
python src/0_clip_demo.py
python src/0_qwen_reproduce.py
python src/1_vqa_evaluation.py
python src/1_vqa_evaluation_with_clip.py
```

## 输出文件说明

### 1. 演示脚本输出
```
results/
├── clip_demo/
│   ├── classification_comparison.png      # 分类可视化对比图
│   ├── classification_result.json         # 分类结果JSON
│   ├── retrieval_comparison.png           # 检索可视化对比图
│   └── retrieval_result.json              # 检索结果JSON
└── qwen_demo/
    ├── vqa_comparison.png                 # VQA可视化对比图
    └── vqa_result.json                    # VQA结果JSON
```

### 2. 评估脚本输出
```
results/
├── vqa_results_api/                       # 基础VQA评估结果
│   ├── all_results.json                   # 所有评估结果
│   ├── category_statistics.png            # 问题类型统计图
│   ├── evaluation_report.json             # 评估报告
│   ├── evaluation_summary.txt             # 评估摘要
│   └── vqa_visualization.png              # 评估可视化图
└── vqa_results_ablation/                  # 消融实验结果
    ├── ablation_comparison.png            # 消融实验对比图
    ├── ablation_report.json               # 消融实验报告
    ├── all_results_clip_disabled.json     # 基线模型结果
    ├── all_results_clip_enabled.json      # CLIP增强结果
    ├── evaluation_report_clip_disabled.json
    ├── evaluation_report_clip_enabled.json
    └── category_statistics.png            # 问题类型统计图
```

