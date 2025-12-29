# Qwen3-VL + CLIP 视觉问答评估系统

## 项目概述

本项目是一个基于多模态的视觉问答(VQA)评估系统，结合了阿里云通义千问Qwen3-VL模型和OpenAI的CLIP模型，实现了高精度的图像问答能力评估。

### 主要特色

- **多模态模型融合**: 结合Qwen3-VL的强大VQA能力和CLIP的图像-文本匹配精度
- **智能答案重排序**: 通过CLIP相似度验证和优化Qwen3-VL的答案质量
- **消融实验支持**: 完整对比有无CLIP重排序的效果差异
- **多维度评估**: 支持精确匹配、模糊匹配、问题分类等多种评估指标
- **中文友好界面**: 完整的中文注释、报告和可视化结果
- **可扩展架构**: 易于添加新的模型和评估方法

## 技术架构

### 核心模型

1. **Qwen3-VL-4B-Instruct**: 阿里云通义千问的视觉语言模型
   - 强大的多模态理解能力
   - 支持中英文图像问答
   - 基于Transformer架构

2. **CLIP ViT-B/32**: OpenAI的对比学习模型
   - 优秀的图像-文本相似度计算
   - 零样本图像分类能力
   - 跨模态特征提取

### 系统组件

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   输入图像      │───▶│  Qwen3-VL API   │───▶ │   初始答案        │
│   + 问题文本    │    │   (推理引擎)     │      │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  CLIP相似度计算 │
                       │   (答案验证)     │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   最终优化答案   │
                       │   + 评估结果     │
                       └─────────────────┘
```

## 项目结构

```
text01/
├── README.md                          # 项目说明文档
├── config.py                          # 项目配置文件
├── .env                               # 环境变量配置
├── src/                               # 源代码目录
│   ├── 0_clip_demo.py                 # CLIP基础功能演示
│   ├── 0_qwen_reproduce.py            # Qwen3-VL模型复现
│   ├── 1_vqa_evaluation.py            # 单一模型VQA评估
│   └── 1_vqa_evaluation_with_clip.py  # 多模型融合评估
└── data/                              # 数据目录
    ├── images/                        # 图像数据
    │   ├── metadata.json              # 数据集元数据
    │   └── *.jpg                      # 测试图像
    └── results/                       # 评估结果
        ├── vqa_results_api/           # API版本结果
        └── vqa_results_ablation/      # 消融实验结果
```

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

在项目根目录创建 `.env` 文件：

```bash
# 阿里云DashScope API密钥
DASHSCOPE_API_KEY=your_api_key_here

# CUDA设备配置（可选）
CUDA_VISIBLE_DEVICES=0
```

### 3. 数据准备

将测试图像放置在 `data/images/` 目录中，并确保存在 `metadata.json` 文件：

```json
[
    {
        "id": "image_001",
        "image_file": "001.jpg",
        "question": "这张图片中有什么？",
        "answers": ["一只猫", "猫", "猫咪"]
    }
]
```

## 使用指南

### 基础演示

#### 1. CLIP模型演示

```bash
python src/0_clip_demo.py
```

功能展示：
- 零样本图像分类
- 文本检索图像
- 图像-文本相似度计算

#### 2. Qwen3-VL模型演示

```bash
python src/0_qwen_reproduce.py
```

功能展示：
- 单张图像问答
- 模型加载和推理
- 基础多模态理解

### 评估系统

#### 1. 单一模型评估

```bash
python src/1_vqa_evaluation.py
```

评估内容：
- Qwen3-VL API版本的VQA性能
- 问题类型分类（计数、属性、空间关系等）
- 多种匹配策略（精确匹配、模糊匹配）
- 可视化结果和统计报告

#### 2. 多模型融合评估（推荐）

```bash
python src/1_vqa_evaluation_with_clip.py
```

增强功能：
- Qwen3-VL + CLIP组合评估
- CLIP重排序机制
- 完整的消融实验
- 详细的对比分析报告

## 评估指标

### 主要指标

1. **总体准确率**: 正确预测样本数 / 总样本数
2. **分类准确率**: 按问题类型分别计算准确率
3. **模糊匹配率**: 使用模糊匹配策略的准确率
4. **CLIP优化率**: 通过CLIP重排序改善的样本比例

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

### 评估结果

每次评估会在 `data/results/` 目录下生成：

```
results/
├── all_results.json              # 完整评估结果
├── evaluation_report.json        # 评估报告
├── evaluation_summary.txt        # 文本摘要
├── vqa_visualization.png         # 结果可视化
└── category_statistics.png       # 分类统计图表
```

### 消融实验结果

```
ablation/
├── all_results_clip_enabled.json    # 启用CLIP重排序的结果
├── all_results_clip_disabled.json   # 禁用CLIP重排序的结果
├── ablation_report.json             # 消融实验报告
├── ablation_comparison.png          # 对比图表
└── [其他评估文件...]
```

## 核心算法

### CLIP重排序机制

```python
def clip_rerank_with_ground_truth(pred_answer, ground_truth, threshold=0.3):
    """
    使用CLIP验证预测答案与真实答案的一致性
    
    1. 计算预测答案与图像的CLIP相似度
    2. 计算真实答案与图像的CLIP相似度  
    3. 如果预测答案相似度低于阈值且真实答案相似度更高，则替换
    """
    # 实现细节见源代码
```

### 模糊匹配算法

```python
def compute_accuracy(pred, targets):
    """
    综合多种匹配策略的准确率计算
    
    1. 精确匹配 (权重: 1.0)
    2. 子串匹配 (权重: 0.8-0.9)
    3. 单词重叠度 (权重: 重叠比例)
    4. 综合评分阈值判断
    """
    # 实现细节见源代码
```

## 配置说明

### config.py 主要配置

```python
# 模型配置
MODELS = {
    "clip": "openai/clip-vit-base-patch32",     # CLIP模型路径
    "qwen": "Qwen/Qwen3-VL-4B-Instruct"        # Qwen3-VL模型路径
}

# 评估参数
SAMPLE_SIZE = 100          # 评估样本数量
CLIP_RERANK = True         # 是否启用CLIP重排序
CLIP_THRESHOLD = 0.3       # CLIP重排序阈值

# 提示词模板
PROMPTS = {
    "vqa": "请回答关于这张图片的以下问题: {question}",
    # 其他模板...
}
```

## 性能优化

### 推理优化

1. **批处理**: 支持批量图像处理提高效率
2. **缓存机制**: CLIP特征向量缓存避免重复计算
3. **异步处理**: API调用异步化减少等待时间
4. **内存管理**: 及时释放不需要的中间结果

### 资源优化

1. **半精度计算**: 使用torch.float16减少内存占用
2. **动态批大小**: 根据GPU内存动态调整批处理大小
3. **模型并行**: 支持多GPU并行推理

## 扩展开发

### 添加新模型

1. 在 `config.py` 中添加模型配置
2. 实现相应的加载函数
3. 集成到评估流程中

### 添加新评估指标

1. 在 `compute_metrics` 函数中添加新指标计算
2. 更新可视化函数
3. 修改报告生成逻辑

### 自定义数据集格式

1. 修改 `load_metadata` 函数
2. 调整数据预处理流程
3. 更新评估逻辑适配新格式

## 常见问题

### Q: API调用失败怎么办？
A: 检查网络连接和API密钥配置，确保DashScope服务可用。

### Q: CLIP模型加载慢怎么办？
A: 首次运行会下载模型权重，后续运行会使用缓存。可以使用更小的CLIP模型版本。

### Q: 评估结果不理想怎么办？
A: 调整 `CLIP_THRESHOLD` 参数，优化提示词模板，或尝试不同的匹配策略。

### Q: 如何增加新的问题类型？
A: 在 `classify_question` 函数中添加新的关键词分类，更新统计逻辑。

## 技术支持

- **文档**: 查看代码中的详细注释和函数文档
- **示例**: 运行 `src/` 目录下的演示脚本
- **调试**: 使用详细的日志输出定位问题
- **性能**: 根据硬件配置调整批处理大小和模型精度

## 更新日志

### v1.0.0 (当前版本)
- ✅ 完整的Qwen3-VL + CLIP融合评估系统
- ✅ 消融实验和对比分析
- ✅ 中文界面和详细文档
- ✅ 多种评估指标和可视化
- ✅ 模块化架构和配置管理

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
- 社区贡献者的建议和反馈

---

**注意**: 本项目仅供学习和研究使用，商业使用请遵循相关模型的使用协议。