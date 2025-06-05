# 电商用户行为分析项目

## 项目简介
基于模拟电商平台用户行为数据，进行用户行为分析和A/B测试，发现关键流失节点并提出优化建议。

## 技术栈
- **Python**: pandas, matplotlib, seaborn, scipy
- **统计分析**: A/B测试、RFM模型、漏斗分析
- **可视化**: matplotlib, seaborn, plotly

## 项目结构
```
├── data/
│   ├── raw/                # 原始数据
│   └── processed/          # 处理后数据
├── notebooks/              # 分析脚本（.py格式，模拟jupyter）
│   ├── 01_数据探索.py
│   ├── 02_用户行为分析.py
│   └── 03_AB测试分析.py
├── src/                    # 核心代码
│   ├── data_generator.py   # 数据生成
│   ├── analysis.py         # 分析函数
│   └── visualization.py    # 可视化函数
├── config.py              # 配置文件
└── requirements.txt       # 依赖包
```

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 生成数据
```python
python src/data_generator.py
```

### 3. 运行分析
```python
# 按顺序运行notebooks中的脚本
python notebooks/01_数据探索.py
python notebooks/02_用户行为分析.py
python notebooks/03_AB测试分析.py
```

## 主要功能

### 📊 用户行为分析
- 用户画像构建（RFM模型）
- 转化漏斗分析
- 用户分群与特征分析
- 流失用户识别

### 🧪 A/B测试
- 实验设计与样本量计算
- 统计显著性检验
- 效果评估与置信区间
- 业务建议制定

## 预期成果
- 识别关键流失节点
- 提供数据驱动的优化建议
- 构建可复用的分析框架 