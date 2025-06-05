# -*- coding: utf-8 -*-
"""
01_数据探索.py
电商用户行为数据探索分析
使用 #%% 分隔代码块，模拟 Jupyter Notebook
"""

#%% 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# 添加src目录到路径
#%%
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
# print(sys.path)
from data_generator import EcommerceDataGenerator
from analysis import EcommerceAnalyzer
from visualization import EcommerceVisualizer
#%%
# 设置pandas显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("📊 电商用户行为数据探索分析")
print("=" * 50)

#%% 步骤1：生成数据（如果不存在）
print("\n🔧 检查并生成数据...")

# 检查数据文件是否存在
data_files = ['data/raw/users.csv', 'data/raw/products.csv', 'data/raw/user_behaviors.csv', 'data/raw/orders.csv']
need_generate = any(not os.path.exists(file) for file in data_files)

if need_generate:
    print("数据文件不存在，正在生成...")
    generator = EcommerceDataGenerator()
    users_df, products_df, behaviors_df, orders_df = generator.generate_all_data()
    print("✅ 数据生成完成！")
else:
    print("✅ 数据文件已存在！")

#%% 步骤2：加载数据
print("\n📂 加载数据...")

# 加载各类数据
users = pd.read_csv('data/raw/users.csv')
products = pd.read_csv('data/raw/products.csv')
behaviors = pd.read_csv('data/raw/user_behaviors.csv')
orders = pd.read_csv('data/raw/orders.csv')

# 转换日期格式
behaviors['behavior_time'] = pd.to_datetime(behaviors['behavior_time'])
orders['order_time'] = pd.to_datetime(orders['order_time'])
users['register_date'] = pd.to_datetime(users['register_date'])

print("数据加载完成！")
print(f"- 用户数据: {len(users)} 条")
print(f"- 商品数据: {len(products)} 条")
print(f"- 行为数据: {len(behaviors)} 条")
print(f"- 订单数据: {len(orders)} 条")

#%% 步骤3：数据概览
print("\n🔍 数据概览")
print("-" * 30)

print("\n【用户数据概览】")
print(users.head())
print(f"\n用户数据形状: {users.shape}")
print(f"数据类型:\n{users.dtypes}")

#%% 用户数据探索性分析
print("\n👥 用户数据探索性分析")
print("-" * 30)

# 基础统计信息
print("【用户基础统计】")
print(f"用户总数: {len(users)}")
print(f"年龄分布: {users['age'].min()}-{users['age'].max()}岁")
print(f"平均年龄: {users['age'].mean():.1f}岁")

# 性别分布
print(f"\n【性别分布】")
gender_counts = users['gender'].value_counts()
print(gender_counts)
print(f"性别比例: {gender_counts.to_dict()}")

# 用户等级分布
print(f"\n【用户等级分布】")
level_counts = users['user_level'].value_counts()
print(level_counts)

# 城市分布（前10）
print(f"\n【城市分布（前10）】")
city_counts = users['city'].value_counts().head(10)
print(city_counts)

#%% 商品数据探索
print("\n🛍️ 商品数据探索")
print("-" * 30)

print("【商品数据概览】")
print(products.head())

print(f"\n【商品基础统计】")
print(f"商品总数: {len(products)}")
print(f"价格范围: {products['price'].min():.2f} - {products['price'].max():.2f}元")
print(f"平均价格: {products['price'].mean():.2f}元")

# 商品类别分布
print(f"\n【商品类别分布】")
category_counts = products['category'].value_counts()
print(category_counts)

# 评分分布
print(f"\n【商品评分分布】")
print(f"评分范围: {products['rating'].min():.2f} - {products['rating'].max():.2f}")
print(f"平均评分: {products['rating'].mean():.2f}")

#%% 用户行为数据探索
print("\n🎯 用户行为数据探索")
print("-" * 30)

print("【行为数据概览】")
print(behaviors.head())

print(f"\n【行为基础统计】")
print(f"行为记录总数: {len(behaviors)}")
print(f"涉及用户数: {behaviors['user_id'].nunique()}")
print(f"涉及商品数: {behaviors['product_id'].nunique()}")

# 行为类型分布
print(f"\n【行为类型分布】")
behavior_counts = behaviors['behavior_type'].value_counts()
print(behavior_counts)
behavior_pct = behaviors['behavior_type'].value_counts(normalize=True) * 100
print(f"\n行为类型占比:")
for behavior, count in behavior_counts.items():
    pct = behavior_pct[behavior]
    print(f"  {behavior}: {count} ({pct:.1f}%)")

# 设备类型分布
print(f"\n【设备类型分布】")
device_counts = behaviors['device_type'].value_counts()
print(device_counts)

# 时间分布分析
print(f"\n【时间分布分析】")
behaviors['hour'] = behaviors['behavior_time'].dt.hour
behaviors['date'] = behaviors['behavior_time'].dt.date

print(f"数据时间范围: {behaviors['behavior_time'].min()} 到 {behaviors['behavior_time'].max()}")
print(f"活跃天数: {behaviors['date'].nunique()} 天")

#%% 订单数据探索
print("\n💰 订单数据探索")
print("-" * 30)

print("【订单数据概览】")
print(orders.head())

print(f"\n【订单基础统计】")
print(f"订单总数: {len(orders)}")
print(f"购买用户数: {orders['user_id'].nunique()}")
print(f"涉及商品数: {orders['product_id'].nunique()}")

# 订单状态分布
print(f"\n【订单状态分布】")
status_counts = orders['status'].value_counts()
print(status_counts)

# 支付方式分布
print(f"\n【支付方式分布】")
payment_counts = orders['payment_method'].value_counts()
print(payment_counts)

# 金额统计
print(f"\n【订单金额统计】")
print(f"订单金额范围: {orders['total_amount'].min():.2f} - {orders['total_amount'].max():.2f}元")
print(f"平均订单金额: {orders['total_amount'].mean():.2f}元")
print(f"订单金额中位数: {orders['total_amount'].median():.2f}元")
print(f"总交易金额: {orders['total_amount'].sum():.2f}元")

#%% 数据质量检查
print("\n🔍 数据质量检查")
print("-" * 30)

def check_data_quality(df, name):
    print(f"\n【{name}数据质量】")
    print(f"数据形状: {df.shape}")
    print(f"缺失值:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "  无缺失值")
    print(f"重复值: {df.duplicated().sum()} 条")
    return missing.sum() == 0 and df.duplicated().sum() == 0

# 检查各表数据质量
users_quality = check_data_quality(users, "用户")
products_quality = check_data_quality(products, "商品")
behaviors_quality = check_data_quality(behaviors, "行为")
orders_quality = check_data_quality(orders, "订单")

print(f"\n✅ 数据质量总评: {'良好' if all([users_quality, products_quality, behaviors_quality, orders_quality]) else '需要清理'}")

#%% 关联性分析
print("\n🔗 数据关联性分析")
print("-" * 30)

# 用户-行为关联
user_behavior_stats = behaviors.groupby('user_id').agg({
    'behavior_type': 'count',
    'product_id': 'nunique'
}).rename(columns={'behavior_type': 'total_behaviors', 'product_id': 'unique_products'})

print("【用户行为统计】")
print(f"平均每用户行为数: {user_behavior_stats['total_behaviors'].mean():.1f}")
print(f"平均每用户浏览商品数: {user_behavior_stats['unique_products'].mean():.1f}")

# 商品-行为关联
product_behavior_stats = behaviors.groupby('product_id').agg({
    'user_id': 'nunique',
    'behavior_type': 'count'
}).rename(columns={'user_id': 'unique_users', 'behavior_type': 'total_behaviors'})

print(f"\n【商品行为统计】")
print(f"平均每商品浏览用户数: {product_behavior_stats['unique_users'].mean():.1f}")
print(f"平均每商品被浏览次数: {product_behavior_stats['total_behaviors'].mean():.1f}")

#%% 初步业务洞察
print("\n💡 初步业务洞察")
print("-" * 30)

# 转化率计算
total_users = behaviors['user_id'].nunique()
browsing_users = behaviors[behaviors['behavior_type'] == 'browse']['user_id'].nunique()
cart_users = behaviors[behaviors['behavior_type'] == 'cart']['user_id'].nunique()
order_users = behaviors[behaviors['behavior_type'] == 'order']['user_id'].nunique()
pay_users = behaviors[behaviors['behavior_type'] == 'pay']['user_id'].nunique()

print("【转化漏斗初步分析】")
print(f"总活跃用户: {total_users}")
print(f"浏览用户: {browsing_users} ({browsing_users/total_users*100:.1f}%)")
print(f"加购用户: {cart_users} ({cart_users/total_users*100:.1f}%)")
print(f"下单用户: {order_users} ({order_users/total_users*100:.1f}%)")
print(f"支付用户: {pay_users} ({pay_users/total_users*100:.1f}%)")

# 用户价值分析
completed_orders = orders[orders['status'] == 'completed']
if len(completed_orders) > 0:
    user_value = completed_orders.groupby('user_id')['total_amount'].sum().describe()
    print(f"\n【用户价值分析】")
    print(f"有效购买用户数: {len(completed_orders['user_id'].unique())}")
    print(f"平均用户价值: {user_value['mean']:.2f}元")
    print(f"用户价值中位数: {user_value['50%']:.2f}元")
    print(f"最高用户价值: {user_value['max']:.2f}元")

#%% 保存探索结果
print("\n💾 保存探索结果")
print("-" * 30)

# 创建输出目录
os.makedirs('data/processed', exist_ok=True)

# 保存处理后的数据
behaviors_processed = behaviors.copy()
behaviors_processed['hour'] = behaviors_processed['behavior_time'].dt.hour
behaviors_processed['date'] = behaviors_processed['behavior_time'].dt.date

# 保存汇总统计
summary_stats = {
    'total_users': len(users),
    'total_products': len(products),
    'total_behaviors': len(behaviors),
    'total_orders': len(orders),
    'conversion_rate': pay_users/total_users*100,
    'avg_order_amount': orders['total_amount'].mean(),
    'total_revenue': orders['total_amount'].sum()
}

# 将字典转换为DataFrame并保存
summary_df = pd.DataFrame([summary_stats])
summary_df.to_csv('data/processed/summary_stats.csv', index=False)

print("✅ 数据探索完成！")
print("📋 探索总结:")
print(f"  - 数据质量: {'良好' if all([users_quality, products_quality, behaviors_quality, orders_quality]) else '需要清理'}")
print(f"  - 用户转化率: {pay_users/total_users*100:.2f}%")
print(f"  - 平均订单金额: {orders['total_amount'].mean():.2f}元")
print(f"  - 数据已保存到 data/processed/ 目录")

#%% 总结
print("\n🎯 下一步计划")
print("-" * 30)
print("1. 深入用户行为分析（RFM模型、用户分群）")
print("2. 转化漏斗详细分析")
print("3. A/B测试设计与分析")
print("4. 制作可视化图表")
print("5. 生成业务报告和建议")

print(f"\n{'='*50}")
print("📊 数据探索分析完成！")
print("👉 请继续运行 02_用户行为分析.py") 