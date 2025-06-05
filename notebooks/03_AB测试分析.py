# -*- coding: utf-8 -*-
"""
03_AB测试分析.py
A/B测试设计与分析，包含实验设计、统计检验、效果评估等
使用 #%% 分隔代码块，模拟 Jupyter Notebook
"""

#%% 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, ttest_ind, chi2_contingency
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from analysis import EcommerceAnalyzer
from visualization import EcommerceVisualizer

# 设置样式
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

print("🧪 A/B测试设计与分析")
print("=" * 50)

#%% 初始化和数据准备
print("\n🔧 初始化分析工具...")

# 创建分析器实例
analyzer = EcommerceAnalyzer()
visualizer = EcommerceVisualizer()

# 加载数据
analyzer.load_data()
print("✅ 数据加载完成！")

# 准备测试数据
users_df = analyzer.users
orders_df = analyzer.orders
behaviors_df = analyzer.behaviors

print(f"数据概况:")
print(f"- 用户数: {len(users_df):,}")
print(f"- 订单数: {len(orders_df):,}")
print(f"- 行为记录: {len(behaviors_df):,}")

#%% A/B测试场景设计
print("\n📋 A/B测试场景设计")
print("-" * 30)

print("【测试场景】新用户优惠券对转化率的影响")
print("- 控制组 (A): 不发放优惠券")
print("- 实验组 (B): 发放10元新用户优惠券")
print("- 主要指标: 转化率、平均订单金额、用户价值")
print("- 次要指标: 复购率、用户满意度")

# 模拟A/B测试分组（基于用户ID的随机分组）
np.random.seed(42)
users_df_test = users_df.copy()

# 筛选新用户（注册时间在最近30天内）
recent_date = users_df['register_date'].max()
cutoff_date = recent_date - pd.Timedelta(days=30)
new_users = users_df_test[users_df_test['register_date'] >= cutoff_date].copy()

if len(new_users) > 100:  # 确保有足够的新用户
    # 随机分组
    new_users['test_group'] = np.random.choice(['A', 'B'], size=len(new_users), p=[0.5, 0.5])
    
    # 统计分组情况
    group_stats = new_users['test_group'].value_counts()
    print(f"\n【分组情况】")
    print(f"控制组 (A): {group_stats.get('A', 0)} 人")
    print(f"实验组 (B): {group_stats.get('B', 0)} 人")
    print(f"总计: {len(new_users)} 人")
    
    # 检查分组平衡性
    print(f"\n【分组平衡性检查】")
    balance_check = new_users.groupby('test_group').agg({
        'age': 'mean',
        'gender': lambda x: (x == 'M').mean(),
        'user_level': lambda x: (x == 'Gold').mean()
    })
    print("各组用户特征对比:")
    print(f"平均年龄: A组 {balance_check.loc['A', 'age']:.1f}岁, B组 {balance_check.loc['B', 'age']:.1f}岁")
    print(f"男性比例: A组 {balance_check.loc['A', 'gender']:.1%}, B组 {balance_check.loc['B', 'gender']:.1%}")
    print(f"Gold用户比例: A组 {balance_check.loc['A', 'user_level']:.1%}, B组 {balance_check.loc['B', 'user_level']:.1%}")
    
    test_ready = True
else:
    print("⚠️ 新用户数量不足，使用所有用户进行模拟测试")
    users_df_test['test_group'] = np.random.choice(['A', 'B'], size=len(users_df_test), p=[0.5, 0.5])
    new_users = users_df_test
    test_ready = True

#%% 样本量计算
print("\n📊 样本量计算")
print("-" * 30)

def calculate_sample_size(baseline_rate, mde, alpha=0.05, power=0.8):
    """
    计算A/B测试所需样本量
    baseline_rate: 基线转化率
    mde: 最小可检测效应 (Minimum Detectable Effect)
    alpha: 显著性水平
    power: 统计功效
    """
    # 使用正态分布近似
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    
    p1 = baseline_rate
    p2 = baseline_rate * (1 + mde)
    
    # 合并标准差
    p_pooled = (p1 + p2) / 2
    
    # 计算样本量
    n = 2 * (z_alpha + z_beta)**2 * p_pooled * (1 - p_pooled) / (p1 - p2)**2
    
    return int(n)

# 估算基线转化率
if test_ready and len(orders_df) > 0:
    # 计算历史转化率
    total_active_users = behaviors_df['user_id'].nunique()
    paying_users = orders_df['user_id'].nunique()
    baseline_conversion_rate = paying_users / total_active_users
    
    print(f"【历史基线数据】")
    print(f"活跃用户数: {total_active_users:,}")
    print(f"付费用户数: {paying_users:,}")
    print(f"基线转化率: {baseline_conversion_rate:.2%}")
    
    # 计算不同效应量下的样本量需求
    print(f"\n【样本量计算】")
    effect_sizes = [0.05, 0.10, 0.15, 0.20]  # 5%, 10%, 15%, 20% 相对提升
    
    for mde in effect_sizes:
        sample_size = calculate_sample_size(baseline_conversion_rate, mde)
        print(f"检测 {mde:.0%} 相对提升: 每组需要 {sample_size:,} 用户")
    
    # 根据实际用户数确定可检测的最小效应
    available_users_per_group = len(new_users) // 2
    print(f"\n实际可用用户: 每组 {available_users_per_group:,} 人")
    
    # 逆向计算可检测的最小效应
    if available_users_per_group > 100:
        detectable_effects = []
        for mde in np.arange(0.01, 0.5, 0.01):
            required_size = calculate_sample_size(baseline_conversion_rate, mde)
            if required_size <= available_users_per_group:
                detectable_effects.append(mde)
        
        if detectable_effects:
            min_detectable = min(detectable_effects)
            print(f"当前样本量可检测最小效应: {min_detectable:.1%}")
        else:
            print("⚠️ 当前样本量可能不足以检测到显著效应")

#%% 模拟实验效果
print("\n🎲 模拟实验效果")
print("-" * 30)

if test_ready:
    # 模拟优惠券对转化率和订单金额的影响
    print("模拟优惠券效果...")
    
    # 获取测试用户的行为和订单数据
    test_user_ids = new_users['user_id'].tolist()
    test_behaviors = behaviors_df[behaviors_df['user_id'].isin(test_user_ids)].copy()
    test_orders = orders_df[orders_df['user_id'].isin(test_user_ids)].copy()
    
    # 合并分组信息
    test_behaviors = test_behaviors.merge(
        new_users[['user_id', 'test_group']], on='user_id', how='left'
    )
    test_orders = test_orders.merge(
        new_users[['user_id', 'test_group']], on='user_id', how='left'
    )
    
    # 模拟B组效果提升（优惠券效果）
    # 假设优惠券使转化率提升15%，平均订单金额提升8%
    conversion_lift = 0.15  # 转化率提升15%
    aov_lift = 0.08        # 平均订单金额提升8%
    
    # 为B组用户增加额外的转化行为（模拟优惠券效果）
    group_b_users = new_users[new_users['test_group'] == 'B']['user_id'].tolist()
    
    # 模拟B组额外的支付行为
    additional_conversions = []
    for user_id in group_b_users:
        if np.random.random() < conversion_lift:  # 15%的概率产生额外转化
            # 随机选择一个商品
            random_product = behaviors_df['product_id'].sample(1).iloc[0]
            additional_conversions.append({
                'user_id': user_id,
                'product_id': random_product,
                'behavior_type': 'pay',
                'behavior_time': datetime.now(),
                'session_id': f'S{np.random.randint(100000, 999999)}',
                'device_type': np.random.choice(['mobile', 'desktop', 'tablet']),
                'test_group': 'B'
            })
    
    if additional_conversions:
        additional_df = pd.DataFrame(additional_conversions)
        test_behaviors = pd.concat([test_behaviors, additional_df], ignore_index=True)
        print(f"为B组增加了 {len(additional_conversions)} 个额外转化")

#%% 实验结果分析
print("\n📈 实验结果分析")
print("-" * 30)

if test_ready and len(test_behaviors) > 0:
    # 按组计算转化指标
    print("【转化率分析】")
    
    # 计算各组的转化率
    group_conversions = test_behaviors[test_behaviors['behavior_type'] == 'pay'].groupby('test_group')['user_id'].nunique()
    group_total_users = test_behaviors.groupby('test_group')['user_id'].nunique()
    
    conversion_rates = {}
    for group in ['A', 'B']:
        conversions = group_conversions.get(group, 0)
        total = group_total_users.get(group, 0)
        rate = conversions / total if total > 0 else 0
        conversion_rates[group] = {
            'conversions': conversions,
            'total_users': total,
            'conversion_rate': rate
        }
        
        print(f"组 {group}:")
        print(f"  总用户数: {total:,}")
        print(f"  转化用户数: {conversions:,}")
        print(f"  转化率: {rate:.2%}")
    
    # 计算提升幅度
    if conversion_rates['A']['conversion_rate'] > 0:
        relative_lift = (conversion_rates['B']['conversion_rate'] - conversion_rates['A']['conversion_rate']) / conversion_rates['A']['conversion_rate']
        absolute_lift = conversion_rates['B']['conversion_rate'] - conversion_rates['A']['conversion_rate']
        
        print(f"\n【效果评估】")
        print(f"绝对提升: {absolute_lift:.2%}")
        print(f"相对提升: {relative_lift:.1%}")

#%% 统计显著性检验
print("\n🔬 统计显著性检验")
print("-" * 30)

if test_ready and len(test_behaviors) > 0:
    # 转化率的卡方检验
    print("【转化率显著性检验 (卡方检验)】")
    
    # 构建列联表
    contingency_table = np.array([
        [conversion_rates['A']['conversions'], conversion_rates['A']['total_users'] - conversion_rates['A']['conversions']],
        [conversion_rates['B']['conversions'], conversion_rates['B']['total_users'] - conversion_rates['B']['conversions']]
    ])
    
    # 执行卡方检验
    chi2, p_value_chi2, dof, expected = chi2_contingency(contingency_table)
    
    print(f"卡方统计量: {chi2:.4f}")
    print(f"p值: {p_value_chi2:.4f}")
    print(f"显著性: {'是' if p_value_chi2 < 0.05 else '否'} (α=0.05)")
    
    # 如果有订单数据，进行订单金额的t检验
    if len(test_orders) > 0:
        print(f"\n【平均订单金额检验 (t检验)】")
        
        group_a_orders = test_orders[test_orders['test_group'] == 'A']['total_amount']
        group_b_orders = test_orders[test_orders['test_group'] == 'B']['total_amount']
        
        if len(group_a_orders) > 0 and len(group_b_orders) > 0:
            # 执行独立样本t检验
            t_stat, p_value_t = ttest_ind(group_a_orders, group_b_orders)
            
            print(f"A组平均订单金额: {group_a_orders.mean():.2f}元 (n={len(group_a_orders)})")
            print(f"B组平均订单金额: {group_b_orders.mean():.2f}元 (n={len(group_b_orders)})")
            print(f"t统计量: {t_stat:.4f}")
            print(f"p值: {p_value_t:.4f}")
            print(f"显著性: {'是' if p_value_t < 0.05 else '否'} (α=0.05)")
            
            # 计算效应量 (Cohen's d)
            pooled_std = np.sqrt(((len(group_a_orders) - 1) * group_a_orders.var() + 
                                 (len(group_b_orders) - 1) * group_b_orders.var()) / 
                                (len(group_a_orders) + len(group_b_orders) - 2))
            cohens_d = (group_b_orders.mean() - group_a_orders.mean()) / pooled_std
            print(f"效应量 (Cohen's d): {cohens_d:.4f}")

#%% 置信区间计算
print("\n📊 置信区间计算")
print("-" * 30)

if test_ready and len(test_behaviors) > 0:
    def calculate_conversion_ci(conversions, total, confidence=0.95):
        """计算转化率的置信区间"""
        if total == 0:
            return 0, 0
        
        p = conversions / total
        z = norm.ppf(1 - (1 - confidence) / 2)
        margin_error = z * np.sqrt(p * (1 - p) / total)
        
        ci_lower = max(0, p - margin_error)
        ci_upper = min(1, p + margin_error)
        
        return ci_lower, ci_upper
    
    print("【转化率95%置信区间】")
    for group in ['A', 'B']:
        rate = conversion_rates[group]['conversion_rate']
        conversions = conversion_rates[group]['conversions']
        total = conversion_rates[group]['total_users']
        
        ci_lower, ci_upper = calculate_conversion_ci(conversions, total)
        
        print(f"组 {group}: {rate:.2%} [{ci_lower:.2%}, {ci_upper:.2%}]")
    
    # 计算提升的置信区间
    if conversion_rates['A']['conversion_rate'] > 0 and conversion_rates['B']['conversion_rate'] > 0:
        print(f"\n【相对提升95%置信区间】")
        
        # 使用Delta方法近似计算相对提升的置信区间
        p_a = conversion_rates['A']['conversion_rate']
        p_b = conversion_rates['B']['conversion_rate']
        n_a = conversion_rates['A']['total_users']
        n_b = conversion_rates['B']['total_users']
        
        # 相对提升的标准误
        relative_lift = (p_b - p_a) / p_a
        se_relative = np.sqrt((p_b * (1 - p_b) / n_b) / p_a**2 + (p_a * (1 - p_a) / n_a * p_b**2) / p_a**4)
        
        z = norm.ppf(0.975)
        ci_lower_rel = relative_lift - z * se_relative
        ci_upper_rel = relative_lift + z * se_relative
        
        print(f"相对提升: {relative_lift:.1%} [{ci_lower_rel:.1%}, {ci_upper_rel:.1%}]")

#%% 高级分析：分层分析
print("\n🔍 分层分析")
print("-" * 30)

if test_ready and len(test_behaviors) > 0:
    print("【按用户特征分层分析】")
    
    # 按性别分层
    print("\n按性别分层:")
    for gender in ['M', 'F']:
        gender_users = new_users[new_users['gender'] == gender]
        if len(gender_users) > 10:  # 确保样本量足够
            gender_behaviors = test_behaviors[test_behaviors['user_id'].isin(gender_users['user_id'])]
            
            # 计算各组转化率
            gender_conversions = gender_behaviors[gender_behaviors['behavior_type'] == 'pay'].groupby('test_group')['user_id'].nunique()
            gender_totals = gender_behaviors.groupby('test_group')['user_id'].nunique()
            
            print(f"  {gender}性用户:")
            for group in ['A', 'B']:
                conversions = gender_conversions.get(group, 0)
                total = gender_totals.get(group, 0)
                rate = conversions / total if total > 0 else 0
                print(f"    组{group}: {rate:.2%} ({conversions}/{total})")
    
    # 按年龄分层
    print("\n按年龄分层:")
    new_users['age_group'] = pd.cut(new_users['age'], bins=[0, 25, 35, 45, 100], labels=['≤25', '26-35', '36-45', '>45'])
    
    for age_group in new_users['age_group'].cat.categories:
        age_users = new_users[new_users['age_group'] == age_group]
        if len(age_users) > 10:
            age_behaviors = test_behaviors[test_behaviors['user_id'].isin(age_users['user_id'])]
            
            age_conversions = age_behaviors[age_behaviors['behavior_type'] == 'pay'].groupby('test_group')['user_id'].nunique()
            age_totals = age_behaviors.groupby('test_group')['user_id'].nunique()
            
            print(f"  {age_group}岁用户:")
            for group in ['A', 'B']:
                conversions = age_conversions.get(group, 0)
                total = age_totals.get(group, 0)
                rate = conversions / total if total > 0 else 0
                print(f"    组{group}: {rate:.2%} ({conversions}/{total})")

#%% 实验持续时间和统计功效分析
print("\n⏱️ 实验持续时间和统计功效分析")
print("-" * 30)

if test_ready:
    def calculate_test_duration(daily_users, required_sample_size):
        """计算实验所需持续时间"""
        return required_sample_size / daily_users
    
    # 估算每日新用户数
    recent_users = users_df[users_df['register_date'] >= cutoff_date]
    days_in_period = (recent_date - cutoff_date).days
    daily_new_users = len(recent_users) / days_in_period if days_in_period > 0 else 1
    
    print(f"【实验持续时间估算】")
    print(f"最近30天日均新用户: {daily_new_users:.1f} 人")
    
    # 计算不同效应量下的实验持续时间
    for mde in [0.10, 0.15, 0.20]:
        required_sample = calculate_sample_size(baseline_conversion_rate, mde)
        duration_days = calculate_test_duration(daily_new_users, required_sample)
        print(f"检测 {mde:.0%} 提升需要: {duration_days:.0f} 天 (每组 {required_sample:,} 用户)")
    
    # 统计功效分析
    print(f"\n【当前实验的统计功效】")
    current_sample_per_group = len(new_users) // 2
    
    def calculate_power(n, baseline_rate, effect_size, alpha=0.05):
        """计算统计功效"""
        p1 = baseline_rate
        p2 = baseline_rate * (1 + effect_size)
        
        # 计算标准化效应量
        pooled_p = (p1 + p2) / 2
        se = np.sqrt(2 * pooled_p * (1 - pooled_p) / n)
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = (abs(p2 - p1) - z_alpha * se) / se
        
        power = norm.cdf(z_beta)
        return power
    
    for effect in [0.10, 0.15, 0.20]:
        power = calculate_power(current_sample_per_group, baseline_conversion_rate, effect)
        print(f"检测 {effect:.0%} 效应的功效: {power:.1%}")

#%% 业务建议和后续行动
print("\n💼 业务建议和后续行动")
print("-" * 30)

if test_ready:
    print("【实验结果总结】")
    
    if 'relative_lift' in locals() and 'p_value_chi2' in locals():
        # 基于实际结果的建议
        is_significant = p_value_chi2 < 0.05
        is_meaningful = abs(relative_lift) > 0.05  # 5%以上的相对改变认为有业务意义
        
        print(f"统计显著性: {'显著' if is_significant else '不显著'}")
        print(f"业务意义: {'有意义' if is_meaningful else '意义不大'}")
        
        if is_significant and is_meaningful:
            if relative_lift > 0:
                print(f"\n✅ 建议全量推广优惠券策略")
                print(f"预期收益: 转化率提升 {relative_lift:.1%}")
            else:
                print(f"\n❌ 建议停止优惠券策略")
                print(f"负面影响: 转化率下降 {abs(relative_lift):.1%}")
        else:
            print(f"\n⚠️ 建议继续观察或重新设计实验")
            if not is_significant:
                print("- 当前结果不具统计显著性，可能需要更长时间或更大样本量")
            if not is_meaningful:
                print("- 效果量较小，考虑调整优惠券面额或策略")
    
    print(f"\n【后续行动计划】")
    print("1. 监控关键指标变化趋势")
    print("2. 分析用户留存和复购行为")
    print("3. 评估优惠券成本效益")
    print("4. 考虑个性化优惠券策略")
    print("5. 设计长期效果跟踪机制")
    
    # 风险评估
    print(f"\n【风险评估】")
    print("• 样本偏差: 仅针对新用户，结果可能不适用于老用户")
    print("• 时间偏差: 实验期间可能受季节性等因素影响")
    print("• 霍桑效应: 用户可能因知道参与实验而改变行为")
    print("• 长期效应: 短期实验可能无法反映长期影响")

#%% 保存实验结果
print("\n💾 保存实验结果")
print("-" * 30)

if test_ready:
    # 创建实验报告
    experiment_report = {
        'experiment_name': '新用户优惠券A/B测试',
        'start_date': datetime.now().strftime('%Y-%m-%d'),
        'test_groups': {
            'A': '控制组（无优惠券）',
            'B': '实验组（10元优惠券）'
        },
        'sample_sizes': {
            'A': int(conversion_rates.get('A', {}).get('total_users', 0)),
            'B': int(conversion_rates.get('B', {}).get('total_users', 0))
        },
        'conversion_rates': {
            'A': float(conversion_rates.get('A', {}).get('conversion_rate', 0)),
            'B': float(conversion_rates.get('B', {}).get('conversion_rate', 0))
        },
        'statistical_significance': bool(p_value_chi2 < 0.05) if 'p_value_chi2' in locals() else None,
        'p_value': float(p_value_chi2) if 'p_value_chi2' in locals() else None,
        'relative_lift': float(relative_lift) if 'relative_lift' in locals() else None,
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 保存实验报告
    import json
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/ab_test_report.json', 'w', encoding='utf-8') as f:
        json.dump(experiment_report, f, ensure_ascii=False, indent=2)
    
    # 保存测试数据
    if 'new_users' in locals():
        new_users.to_csv('data/processed/ab_test_users.csv', index=False)
    
    print("✅ A/B测试分析完成！")
    print("\n📋 实验总结:")
    print(f"  - 测试用户: {len(new_users) if 'new_users' in locals() else 0:,} 人")
    if 'conversion_rates' in locals():
        print(f"  - A组转化率: {conversion_rates.get('A', {}).get('conversion_rate', 0):.2%}")
        print(f"  - B组转化率: {conversion_rates.get('B', {}).get('conversion_rate', 0):.2%}")
    if 'relative_lift' in locals():
        print(f"  - 相对提升: {relative_lift:.1%}")
    if 'p_value_chi2' in locals():
        print(f"  - 统计显著性: {'显著' if p_value_chi2 < 0.05 else '不显著'} (p={p_value_chi2:.4f})")
    print(f"  - 实验报告已保存到 data/processed/ 目录")

#%% 实验学习和改进
print("\n🎓 实验学习和改进")
print("-" * 30)

print("【本次实验的学习要点】")
print("1. 样本量计算的重要性 - 确保有足够的统计功效")
print("2. 分组平衡性检查 - 避免混淆变量影响结果")
print("3. 多重指标分析 - 不仅看转化率，还要看订单金额等")
print("4. 分层分析价值 - 不同用户群体可能有不同反应")
print("5. 置信区间解读 - 点估计要结合区间估计看")

print(f"\n【改进建议】")
print("• 增加更多业务指标：用户留存、LTV、满意度等")
print("• 考虑多变量测试：同时测试优惠券面额、文案等")
print("• 实施序贯分析：动态调整实验持续时间")
print("• 建立实验平台：标准化A/B测试流程")
print("• 长期跟踪机制：评估实验的长期业务影响")

print(f"\n{'='*50}")
print("🧪 A/B测试分析完成！")
print("🎉 电商用户行为分析项目全部完成！")
print("\n📊 项目成果:")
print("  ✅ 数据探索和质量评估")
print("  ✅ 用户行为深度分析")
print("  ✅ RFM用户价值分析") 
print("  ✅ 转化漏斗分析")
print("  ✅ A/B测试设计和分析")
print("  ✅ 业务洞察和建议")
print("\n🚀 恭喜您完成了一个完整的数据分析项目！") 
# %%
