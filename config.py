# -*- coding: utf-8 -*-
"""
项目配置文件
"""

# 数据生成参数
DATA_CONFIG = {
    'n_users': 10000,           # 用户数量
    'n_products': 500,          # 商品数量
    'n_categories': 20,         # 商品类别数量
    'date_range': 90,           # 数据时间范围（天）
    'random_seed': 42           # 随机种子
}

# 分析参数
ANALYSIS_CONFIG = {
    'funnel_steps': ['browse', 'cart', 'order', 'pay'],  # 转化漏斗步骤
    'rfm_quantiles': 5,         # RFM分析分位数
    'ab_test_ratio': 0.5,       # A/B测试分组比例
    'significance_level': 0.05   # 显著性水平
}

# 文件路径
PATHS = {
    'raw_data': 'data/raw/',
    'processed_data': 'data/processed/',
    'outputs': 'outputs/'
}

# 可视化设置
PLOT_CONFIG = {
    'figsize': (12, 8),
    'style': 'whitegrid',
    'palette': 'Set2',
    'dpi': 100
} 