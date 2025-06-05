# -*- coding: utf-8 -*-
"""
02_用户行为分析.py
深入的用户行为分析，包含RFM模型、转化漏斗、用户分群等
使用 #%% 分隔代码块，模拟 Jupyter Notebook
"""

#%% 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from datetime import datetime, timedelta

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from analysis import EcommerceAnalyzer
from visualization import EcommerceVisualizer

# 设置样式
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

print("🎯 电商用户行为深度分析")
print("=" * 50)

#%% 初始化分析器和可视化器
print("\n🔧 初始化分析工具...")

# 创建分析器和可视化器实例
analyzer = EcommerceAnalyzer()
visualizer = EcommerceVisualizer()

# 加载数据
analyzer.load_data()
print("✅ 数据加载完成！")

#%% 基础统计分析
print("\n📊 基础统计分析")
print("-" * 30)

# 获取基础统计
basic_stats = analyzer.basic_stats()
print("【平台核心指标】")
for metric, value in basic_stats.items():
    if '金额' in metric:
        print(f"{metric}: {value:,.2f} 元")
    elif '数' in metric:
        print(f"{metric}: {value:,}")
    else:
        print(f"{metric}: {value}")

# 计算关键业务指标
conversion_rate = (basic_stats['购买用户数'] / basic_stats['活跃用户数']) * 100
print(f"\n🎯 关键指标:")
print(f"用户转化率: {conversion_rate:.2f}%")
print(f"人均订单金额: {basic_stats['平均订单金额']:.2f}元")
print(f"活跃用户占比: {(basic_stats['活跃用户数'] / basic_stats['用户总数']) * 100:.1f}%")

#%% RFM用户价值分析
print("\n💎 RFM用户价值分析")
print("-" * 30)

# 执行RFM分析
rfm_results = analyzer.rfm_analysis()
print("✅ RFM分析完成！")

# RFM分布统计
print("\n【RFM指标分布】")
print(f"最近购买时间 (Recency):")
print(f"  平均: {rfm_results['recency'].mean():.1f} 天")
print(f"  中位数: {rfm_results['recency'].median():.1f} 天")
print(f"  范围: {rfm_results['recency'].min()}-{rfm_results['recency'].max()} 天")

print(f"\n购买频次 (Frequency):")
print(f"  平均: {rfm_results['frequency'].mean():.1f} 次")
print(f"  中位数: {rfm_results['frequency'].median():.1f} 次")
print(f"  最高: {rfm_results['frequency'].max()} 次")

print(f"\n消费金额 (Monetary):")
print(f"  平均: {rfm_results['monetary'].mean():.2f} 元")
print(f"  中位数: {rfm_results['monetary'].median():.2f} 元")
print(f"  最高: {rfm_results['monetary'].max():.2f} 元")

# 用户分群分析
print("\n【用户分群结果】")
segment_analysis = rfm_results['customer_segment'].value_counts()
segment_pct = rfm_results['customer_segment'].value_counts(normalize=True) * 100

for segment, count in segment_analysis.items():
    pct = segment_pct[segment]
    avg_monetary = rfm_results[rfm_results['customer_segment'] == segment]['monetary'].mean()
    print(f"{segment}: {count} 人 ({pct:.1f}%) - 平均消费 {avg_monetary:.2f}元")

# 保存RFM结果
os.makedirs('data/processed', exist_ok=True)
rfm_results.to_csv('data/processed/rfm_analysis.csv', index=False)
print(f"\n💾 RFM分析结果已保存")

#%% 转化漏斗分析
print("\n🔄 转化漏斗分析")
print("-" * 30)

# 执行漏斗分析
funnel_results = analyzer.funnel_analysis()
print("✅ 漏斗分析完成！")

# 显示漏斗结果
print("\n【转化漏斗详情】")
for _, row in funnel_results.iterrows():
    step_name = row['step_name']
    user_count = row['user_count']
    conversion_rate = row['conversion_rate']
    step_conversion = row['step_conversion_rate']
    
    print(f"{step_name}:")
    print(f"  用户数: {user_count:,}")
    print(f"  总转化率: {conversion_rate:.2f}%")
    if step_conversion > 0:
        print(f"  步骤转化率: {step_conversion:.2f}%")
    print()

# 找出最大流失点
step_conversions = funnel_results['step_conversion_rate'][1:]  # 排除第一步
if len(step_conversions) > 0:
    min_conversion_idx = step_conversions.idxmin()
    worst_step = funnel_results.iloc[min_conversion_idx]['step_name']
    worst_rate = funnel_results.iloc[min_conversion_idx]['step_conversion_rate']
    print(f"🚨 关键流失点: {worst_step} (转化率仅 {worst_rate:.1f}%)")

# 保存漏斗结果
funnel_results.to_csv('data/processed/funnel_analysis.csv', index=False)

#%% 用户行为模式分析
print("\n📱 用户行为模式分析")
print("-" * 30)

# 执行用户行为分析
behavior_analysis = analyzer.user_behavior_analysis()
print("✅ 行为模式分析完成！")

# 设备使用分析
print("\n【设备使用分析】")
device_data = behavior_analysis['device_analysis']
for device, data in device_data.iterrows():
    unique_users = data['unique_users']
    total_behaviors = data['total_behaviors']
    avg_behaviors = total_behaviors / unique_users
    print(f"{device}:")
    print(f"  独立用户: {unique_users:,}")
    print(f"  行为总数: {total_behaviors:,}")
    print(f"  人均行为: {avg_behaviors:.1f} 次")

# 时间分布分析
print("\n【用户活跃时间分析】")
hourly_data = behavior_analysis['hourly_analysis']
peak_hour = hourly_data.idxmax()
peak_count = hourly_data.max()
print(f"最活跃时段: {peak_hour}:00 ({peak_count:,} 次行为)")

# 统计时段分布
morning = hourly_data[6:12].sum()    # 6-12点
afternoon = hourly_data[12:18].sum() # 12-18点
evening = hourly_data[18:24].sum()   # 18-24点
night = hourly_data[0:6].sum()       # 0-6点

total_behaviors = morning + afternoon + evening + night
print(f"时段分布:")
print(f"  上午 (6-12点): {morning:,} ({morning/total_behaviors*100:.1f}%)")
print(f"  下午 (12-18点): {afternoon:,} ({afternoon/total_behaviors*100:.1f}%)")
print(f"  晚上 (18-24点): {evening:,} ({evening/total_behaviors*100:.1f}%)")
print(f"  深夜 (0-6点): {night:,} ({night/total_behaviors*100:.1f}%)")

# 用户等级行为分析
print("\n【用户等级行为对比】")
level_behavior = behavior_analysis['user_level_analysis']
for level in level_behavior.index:
    total_level_behaviors = level_behavior.loc[level].sum()
    print(f"{level}用户:")
    for behavior_type in level_behavior.columns:
        count = level_behavior.loc[level, behavior_type]
        pct = count / total_level_behaviors * 100 if total_level_behaviors > 0 else 0
        print(f"  {behavior_type}: {count} ({pct:.1f}%)")

#%% 用户留存分析
print("\n📈 用户留存分析")
print("-" * 30)

try:
    # 执行留存分析
    cohort_results = analyzer.cohort_analysis()
    print("✅ 留存分析完成！")
    
    # 分析留存趋势
    if not cohort_results.empty:
        print("\n【留存率趋势】")
        # 计算各月度的平均留存率
        for month in range(min(6, cohort_results.shape[1])):
            if month in cohort_results.columns:
                avg_retention = cohort_results[month].mean()
                print(f"第{month}个月留存率: {avg_retention:.2%}")
        
        # 保存留存结果
        cohort_results.to_csv('data/processed/cohort_analysis.csv')
        print(f"\n💾 留存分析结果已保存")
    else:
        print("⚠️  留存数据不足，跳过分析")
        
except Exception as e:
    print(f"⚠️  留存分析遇到问题: {e}")

#%% 用户生命周期价值分析
print("\n💰 用户生命周期价值分析")
print("-" * 30)

# 计算用户LTV (简化版)
orders_df = analyzer.orders
users_df = analyzer.users

# 合并订单和用户数据
user_orders = orders_df.merge(users_df[['user_id', 'register_date']], on='user_id')
user_orders['order_time'] = pd.to_datetime(user_orders['order_time'])
user_orders['register_date'] = pd.to_datetime(user_orders['register_date'])

# 计算用户的生命周期指标
user_ltv = user_orders.groupby('user_id').agg({
    'total_amount': ['sum', 'mean', 'count'],
    'order_time': ['min', 'max']
}).reset_index()

# 扁平化列名
user_ltv.columns = ['user_id', 'total_spent', 'avg_order_value', 'order_count', 'first_order', 'last_order']

# 计算生命周期天数
user_ltv['lifecycle_days'] = (user_ltv['last_order'] - user_ltv['first_order']).dt.days + 1

# 过滤有效数据
valid_users = user_ltv[user_ltv['order_count'] > 0]

if len(valid_users) > 0:
    print("【用户价值分布】")
    print(f"购买用户总数: {len(valid_users)}")
    print(f"平均订单价值: {valid_users['avg_order_value'].mean():.2f}元")
    print(f"平均购买次数: {valid_users['order_count'].mean():.1f}次")
    print(f"平均生命周期: {valid_users['lifecycle_days'].mean():.1f}天")
    print(f"平均用户价值: {valid_users['total_spent'].mean():.2f}元")
    
    # 价值分层
    high_value_threshold = valid_users['total_spent'].quantile(0.8)
    medium_value_threshold = valid_users['total_spent'].quantile(0.5)
    
    high_value_users = len(valid_users[valid_users['total_spent'] >= high_value_threshold])
    medium_value_users = len(valid_users[
        (valid_users['total_spent'] >= medium_value_threshold) & 
        (valid_users['total_spent'] < high_value_threshold)
    ])
    low_value_users = len(valid_users[valid_users['total_spent'] < medium_value_threshold])
    
    print(f"\n【用户价值分层】")
    print(f"高价值用户 (>{high_value_threshold:.0f}元): {high_value_users} ({high_value_users/len(valid_users)*100:.1f}%)")
    print(f"中价值用户 ({medium_value_threshold:.0f}-{high_value_threshold:.0f}元): {medium_value_users} ({medium_value_users/len(valid_users)*100:.1f}%)")
    print(f"低价值用户 (<{medium_value_threshold:.0f}元): {low_value_users} ({low_value_users/len(valid_users)*100:.1f}%)")
    
    # 保存LTV分析
    valid_users.to_csv('data/processed/user_ltv_analysis.csv', index=False)

#%% 商品分析
print("\n🛍️ 商品分析")
print("-" * 30)

# 商品受欢迎程度分析
behaviors_df = analyzer.behaviors
products_df = analyzer.products

# 商品行为统计
product_stats = behaviors_df.groupby('product_id').agg({
    'user_id': 'nunique',
    'behavior_type': 'count'
}).rename(columns={'user_id': 'unique_users', 'behavior_type': 'total_interactions'})

# 合并商品信息
product_analysis = product_stats.merge(products_df, on='product_id', how='left')

# 分析最受欢迎的商品类别
category_popularity = product_analysis.groupby('category').agg({
    'unique_users': 'sum',
    'total_interactions': 'sum',
    'product_id': 'count'
}).rename(columns={'product_id': 'product_count'})

category_popularity['avg_interactions_per_product'] = (
    category_popularity['total_interactions'] / category_popularity['product_count']
)

print("【商品类别受欢迎程度 (前10)】")
top_categories = category_popularity.sort_values('unique_users', ascending=False).head(10)
for category, data in top_categories.iterrows():
    print(f"{category}:")
    print(f"  独立用户: {data['unique_users']:,}")
    print(f"  总交互: {data['total_interactions']:,}")
    print(f"  商品数量: {data['product_count']}")
    print(f"  平均每商品交互: {data['avg_interactions_per_product']:.1f}")

#%% 生成业务洞察
print("\n💡 业务洞察与建议")
print("-" * 30)

# 使用分析器生成洞察
insights = analyzer.generate_insights()

print("【数据驱动的业务洞察】")
for i, insight in enumerate(insights, 1):
    print(f"{i}. {insight}")

# 补充详细建议
print(f"\n【具体优化建议】")

# 基于RFM分析的建议
champion_ratio = (rfm_results['customer_segment'] == '冠军用户').mean() * 100
churn_ratio = (rfm_results['customer_segment'] == '流失用户').mean() * 100

if champion_ratio < 15:
    print("• 冠军用户占比偏低，建议加强用户激励和忠诚度计划")

if churn_ratio > 20:
    print("• 流失用户占比较高，建议实施召回策略和个性化推荐")

# 基于漏斗分析的建议
min_conversion_step = funnel_results.loc[funnel_results['step_conversion_rate'].idxmin()]
if min_conversion_step['step_conversion_rate'] < 30:
    step_name = min_conversion_step['step_name']
    print(f"• {step_name}环节转化率过低，建议优化用户体验和流程设计")

# 基于时间分析的建议
if evening > afternoon:
    print("• 用户晚间活跃度高，建议在18-24点增加营销活动和推送")

#%% 保存分析结果
print("\n💾 保存分析结果")
print("-" * 30)

# 创建综合报告
report_data = {
    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'total_users': len(analyzer.users),
    'active_users': analyzer.behaviors['user_id'].nunique(),
    'conversion_rate': conversion_rate,
    'champion_users_pct': champion_ratio,
    'churn_users_pct': churn_ratio,
    'peak_hour': int(peak_hour),
    'avg_order_value': basic_stats['平均订单金额'],
    'key_insights': insights
}

# 保存为JSON格式
import json
with open('data/processed/behavior_analysis_report.json', 'w', encoding='utf-8') as f:
    json.dump(report_data, f, ensure_ascii=False, indent=2)

print("✅ 用户行为分析完成！")
print("\n📋 分析总结:")
print(f"  - RFM用户分群: 冠军用户 {champion_ratio:.1f}%, 流失用户 {churn_ratio:.1f}%")
print(f"  - 转化率: {conversion_rate:.2f}%")
print(f"  - 用户最活跃时段: {peak_hour}:00")
print(f"  - 平均订单金额: {basic_stats['平均订单金额']:.2f}元")
print(f"  - 分析报告已保存到 data/processed/ 目录")

#%% 下一步计划
print("\n🎯 下一步计划")
print("-" * 30)
print("1. A/B测试设计和分析")
print("2. 用户细分策略制定")
print("3. 个性化推荐方案")
print("4. 制作数据可视化Dashboard")
print("5. 撰写最终业务报告")

print(f"\n{'='*50}")
print("🎯 用户行为分析完成！")
print("👉 请继续运行 03_AB测试分析.py") 