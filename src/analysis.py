# -*- coding: utf-8 -*-
"""
数据分析函数库
包含RFM分析、漏斗分析、A/B测试等核心分析方法
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class EcommerceAnalyzer:
    def __init__(self):
        self.reference_date = datetime.now()
    
    def load_data(self, data_path='data/raw/'):
        """加载数据"""
        self.users = pd.read_csv(f'{data_path}users.csv')
        self.products = pd.read_csv(f'{data_path}products.csv')
        self.behaviors = pd.read_csv(f'{data_path}user_behaviors.csv')
        self.orders = pd.read_csv(f'{data_path}orders.csv')
        
        # 转换日期格式
        self.behaviors['behavior_time'] = pd.to_datetime(self.behaviors['behavior_time'])
        self.orders['order_time'] = pd.to_datetime(self.orders['order_time'])
        self.users['register_date'] = pd.to_datetime(self.users['register_date'])
        
        print("数据加载完成！")
        return self
    
    def basic_stats(self):
        """基础数据统计"""
        stats_dict = {
            '用户总数': len(self.users),
            '商品总数': len(self.products),
            '行为记录数': len(self.behaviors),
            '订单总数': len(self.orders),
            '活跃用户数': self.behaviors['user_id'].nunique(),
            '购买用户数': self.orders['user_id'].nunique(),
            '总交易金额': self.orders['total_amount'].sum(),
            '平均订单金额': self.orders['total_amount'].mean()
        }
        return pd.Series(stats_dict)
    
    def rfm_analysis(self):
        """RFM分析"""
        print("执行RFM分析...")
        
        # 计算RFM指标
        rfm_data = []
        
        for user_id in self.orders['user_id'].unique():
            user_orders = self.orders[self.orders['user_id'] == user_id]
            
            # Recency: 最近一次购买距今天数
            last_order_date = user_orders['order_time'].max()
            recency = (self.reference_date - last_order_date).days
            
            # Frequency: 购买频次
            frequency = len(user_orders)
            
            # Monetary: 购买金额
            monetary = user_orders['total_amount'].sum()
            
            rfm_data.append({
                'user_id': user_id,
                'recency': recency,
                'frequency': frequency,
                'monetary': monetary
            })
        
        rfm_df = pd.DataFrame(rfm_data)
        
        # RFM分组（5分位数）
        rfm_df['r_score'] = pd.qcut(rfm_df['recency'], 5, labels=[5,4,3,2,1])
        rfm_df['f_score'] = pd.qcut(rfm_df['frequency'], 5, labels=[1,2,3,4,5])
        rfm_df['m_score'] = pd.qcut(rfm_df['monetary'], 5, labels=[1,2,3,4,5])
        
        # 组合RFM分数
        rfm_df['rfm_score'] = rfm_df['r_score'].astype(str) + \
                             rfm_df['f_score'].astype(str) + \
                             rfm_df['m_score'].astype(str)
        
        # 用户分群
        def categorize_customers(row):
            if row['rfm_score'] in ['555', '554', '544', '545', '454', '455', '445']:
                return '冠军用户'
            elif row['rfm_score'] in ['543', '444', '435', '355', '354', '345', '344', '335']:
                return '忠实用户'
            elif row['rfm_score'] in ['512', '511', '422', '421', '412', '411', '311']:
                return '新用户'
            elif row['rfm_score'] in ['155', '154', '144', '214', '215', '115', '114']:
                return '大客户'
            elif row['rfm_score'] in ['331', '321', '231', '241', '251']:
                return '需要关怀的用户'
            elif row['rfm_score'] in ['111', '112', '121', '131', '141', '151']:
                return '流失用户'
            else:
                return '其他'
        
        rfm_df['customer_segment'] = rfm_df.apply(categorize_customers, axis=1)
        
        return rfm_df
    
    def funnel_analysis(self):
        """转化漏斗分析"""
        print("执行转化漏斗分析...")
        
        # 计算各步骤的用户数
        funnel_steps = ['browse', 'cart', 'order', 'pay']
        funnel_data = []
        
        total_users = self.behaviors['user_id'].nunique()
        
        for step in funnel_steps:
            step_users = self.behaviors[self.behaviors['behavior_type'] == step]['user_id'].nunique()
            conversion_rate = step_users / total_users * 100
            
            funnel_data.append({
                'step': step,
                'step_name': {'browse': '浏览', 'cart': '加购物车', 'order': '下单', 'pay': '支付'}[step],
                'user_count': step_users,
                'conversion_rate': conversion_rate
            })
        
        funnel_df = pd.DataFrame(funnel_data)
        
        # 计算步骤间转化率
        funnel_df['step_conversion_rate'] = 0.0
        for i in range(1, len(funnel_df)):
            if funnel_df.iloc[i-1]['user_count'] > 0:
                step_rate = funnel_df.iloc[i]['user_count'] / funnel_df.iloc[i-1]['user_count'] * 100
                funnel_df.iloc[i, funnel_df.columns.get_loc('step_conversion_rate')] = step_rate
        
        return funnel_df
    
    def user_behavior_analysis(self):
        """用户行为分析"""
        print("执行用户行为分析...")
        
        # 按设备类型分析
        device_analysis = self.behaviors.groupby('device_type').agg({
            'user_id': 'nunique',
            'behavior_type': 'count'
        }).rename(columns={'user_id': 'unique_users', 'behavior_type': 'total_behaviors'})
        
        # 按时间分析（小时）
        self.behaviors['hour'] = self.behaviors['behavior_time'].dt.hour
        hourly_analysis = self.behaviors.groupby('hour').size()
        
        # 按用户等级分析
        user_level_behavior = self.behaviors.merge(
            self.users[['user_id', 'user_level']], on='user_id'
        ).groupby(['user_level', 'behavior_type']).size().unstack(fill_value=0)
        
        return {
            'device_analysis': device_analysis,
            'hourly_analysis': hourly_analysis,
            'user_level_analysis': user_level_behavior
        }
    
    def cohort_analysis(self):
        """用户留存分析（简化版）"""
        print("执行用户留存分析...")
        
        # 以注册月为队列
        self.users['register_month'] = self.users['register_date'].dt.to_period('M')
        
        # 获取每个用户的活跃月份
        self.behaviors['behavior_month'] = self.behaviors['behavior_time'].dt.to_period('M')
        
        # 合并数据
        user_behavior_cohort = self.behaviors.merge(
            self.users[['user_id', 'register_month']], on='user_id'
        )
        
        # 计算每个队列在各月的活跃用户数
        cohort_data = user_behavior_cohort.groupby(['register_month', 'behavior_month'])['user_id'].nunique().reset_index()
        
        # 计算月份差异
        cohort_data['months_diff'] = (
            cohort_data['behavior_month'] - cohort_data['register_month']
        ).apply(lambda x: x.n)
        
        # 构建留存表
        cohort_table = cohort_data.pivot(index='register_month', 
                                       columns='months_diff', 
                                       values='user_id')
        
        # 计算留存率
        cohort_sizes = self.users.groupby('register_month')['user_id'].nunique()
        cohort_table = cohort_table.divide(cohort_sizes, axis=0)
        
        return cohort_table
    
    def ab_test_analysis(self, metric_column='total_amount', test_column='user_level'):
        """A/B测试分析（以用户等级为例）"""
        print("执行A/B测试分析...")
        
        # 创建测试组（简化：以Bronze vs Gold为例）
        group_a = self.orders.merge(self.users, on='user_id')
        group_a = group_a[group_a['user_level'] == 'Bronze']
        
        group_b = self.orders.merge(self.users, on='user_id')
        group_b = group_b[group_b['user_level'] == 'Gold']
        
        # 提取指标数据
        metric_a = group_a[metric_column].dropna()
        metric_b = group_b[metric_column].dropna()
        
        # 统计检验
        t_stat, p_value = stats.ttest_ind(metric_a, metric_b)
        
        # 计算效应量（Cohen's d）
        pooled_std = np.sqrt(((len(metric_a) - 1) * metric_a.var() + 
                             (len(metric_b) - 1) * metric_b.var()) / 
                            (len(metric_a) + len(metric_b) - 2))
        cohens_d = (metric_b.mean() - metric_a.mean()) / pooled_std
        
        # 置信区间
        se_diff = pooled_std * np.sqrt(1/len(metric_a) + 1/len(metric_b))
        ci_lower = (metric_b.mean() - metric_a.mean()) - 1.96 * se_diff
        ci_upper = (metric_b.mean() - metric_a.mean()) + 1.96 * se_diff
        
        results = {
            'group_a_mean': metric_a.mean(),
            'group_b_mean': metric_b.mean(),
            'group_a_size': len(metric_a),
            'group_b_size': len(metric_b),
            'difference': metric_b.mean() - metric_a.mean(),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': p_value < 0.05
        }
        
        return results
    
    def generate_insights(self):
        """生成业务洞察"""
        insights = []
        
        # 基础统计洞察
        basic = self.basic_stats()
        conversion_rate = (basic['购买用户数'] / basic['活跃用户数']) * 100
        insights.append(f"整体转化率为 {conversion_rate:.2f}%")
        
        # 漏斗分析洞察
        funnel = self.funnel_analysis()
        max_drop_idx = funnel['step_conversion_rate'][1:].idxmin()
        max_drop_step = funnel.iloc[max_drop_idx]['step_name']
        insights.append(f"最大流失发生在 {max_drop_step} 步骤")
        
        # RFM分析洞察
        rfm = self.rfm_analysis()
        champion_pct = (rfm['customer_segment'] == '冠军用户').sum() / len(rfm) * 100
        churn_pct = (rfm['customer_segment'] == '流失用户').sum() / len(rfm) * 100
        insights.append(f"冠军用户占比 {champion_pct:.1f}%，流失用户占比 {churn_pct:.1f}%")
        
        return insights

# 示例使用
if __name__ == "__main__":
    analyzer = EcommerceAnalyzer()
    analyzer.load_data()
    
    print("=== 基础统计 ===")
    print(analyzer.basic_stats())
    
    print("\n=== 业务洞察 ===")
    insights = analyzer.generate_insights()
    for insight in insights:
        print(f"• {insight}") 