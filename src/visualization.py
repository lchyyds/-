# -*- coding: utf-8 -*-
"""
数据可视化函数库
包含各种图表生成函数
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 安全导入plotly - 解决numpy 2.0兼容性问题
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except Exception:
    # 创建占位符，避免后续代码报错
    class PlotlyMock:
        def __getattr__(self, name):
            def mock_func(*args, **kwargs):
                # 返回matplotlib figure作为替代
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, f'Plotly功能不可用\n使用matplotlib替代', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title('Plotly功能替代')
                return fig
            return mock_func
    
    px = PlotlyMock()
    go = PlotlyMock()
    make_subplots = PlotlyMock()
    PLOTLY_AVAILABLE = False

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class EcommerceVisualizer:
    def __init__(self, style='whitegrid', palette='Set2', figsize=(12, 8)):
        self.style = style
        self.palette = palette
        self.figsize = figsize
        sns.set_style(style)
        sns.set_palette(palette)
    
    def plot_basic_stats(self, stats_data):
        """绘制基础统计图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('电商平台基础统计概览', fontsize=16, fontweight='bold')
        
        # 用户与商品统计
        user_product_stats = stats_data[['用户总数', '商品总数', '活跃用户数', '购买用户数']]
        axes[0, 0].bar(range(len(user_product_stats)), user_product_stats.values)
        axes[0, 0].set_xticks(range(len(user_product_stats)))
        axes[0, 0].set_xticklabels(user_product_stats.index, rotation=45)
        axes[0, 0].set_title('用户与商品统计')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 行为与订单统计
        behavior_order_stats = stats_data[['行为记录数', '订单总数']]
        axes[0, 1].bar(range(len(behavior_order_stats)), behavior_order_stats.values, color='orange')
        axes[0, 1].set_xticks(range(len(behavior_order_stats)))
        axes[0, 1].set_xticklabels(behavior_order_stats.index)
        axes[0, 1].set_title('行为与订单统计')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 转化率
        conversion_rate = (stats_data['购买用户数'] / stats_data['活跃用户数']) * 100
        axes[1, 0].pie([conversion_rate, 100-conversion_rate], 
                      labels=[f'购买用户 {conversion_rate:.1f}%', f'未购买用户 {100-conversion_rate:.1f}%'],
                      autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('用户转化率')
        
        # 平均订单金额
        avg_order = stats_data['平均订单金额']
        total_amount = stats_data['总交易金额']
        axes[1, 1].bar(['平均订单金额', '总交易金额'], [avg_order, total_amount/1000], 
                      color=['skyblue', 'lightcoral'])
        axes[1, 1].set_ylabel('金额（元 / 千元）')
        axes[1, 1].set_title('交易金额统计')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_funnel(self, funnel_data):
        """绘制转化漏斗图"""
        fig = go.Figure()
        
        # 漏斗图
        fig.add_trace(go.Funnel(
            y=funnel_data['step_name'],
            x=funnel_data['user_count'],
            textinfo="value+percent initial",
            textfont=dict(size=14),
            connector={"line": {"color": "royalblue", "dash": "dot", "width": 3}},
            marker={"color": ["deepskyblue", "lightsalmon", "tan", "teal"]}
        ))
        
        fig.update_layout(
            title="用户转化漏斗分析",
            font=dict(size=14),
            width=800,
            height=600
        )
        
        return fig
    
    def plot_rfm_analysis(self, rfm_data):
        """绘制RFM分析图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RFM用户分析', fontsize=16, fontweight='bold')
        
        # RFM分布直方图
        axes[0, 0].hist(rfm_data['recency'], bins=30, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('最近购买时间分布（天）')
        axes[0, 0].set_xlabel('天数')
        axes[0, 0].set_ylabel('用户数')
        
        axes[0, 1].hist(rfm_data['frequency'], bins=20, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('购买频次分布')
        axes[0, 1].set_xlabel('购买次数')
        axes[0, 1].set_ylabel('用户数')
        
        # 用户分群饼图
        segment_counts = rfm_data['customer_segment'].value_counts()
        axes[1, 0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('用户分群分布')
        
        # 消费金额分布
        axes[1, 1].hist(rfm_data['monetary'], bins=30, alpha=0.7, color='salmon')
        axes[1, 1].set_title('消费金额分布')
        axes[1, 1].set_xlabel('金额（元）')
        axes[1, 1].set_ylabel('用户数')
        
        plt.tight_layout()
        return fig
    
    def plot_behavior_analysis(self, behavior_data):
        """绘制用户行为分析图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('用户行为分析', fontsize=16, fontweight='bold')
        
        # 设备类型分析
        device_analysis = behavior_data['device_analysis']
        axes[0, 0].bar(device_analysis.index, device_analysis['unique_users'])
        axes[0, 0].set_title('不同设备类型用户数')
        axes[0, 0].set_ylabel('独立用户数')
        
        # 行为总数按设备
        axes[0, 1].bar(device_analysis.index, device_analysis['total_behaviors'], color='orange')
        axes[0, 1].set_title('不同设备类型行为总数')
        axes[0, 1].set_ylabel('行为总数')
        
        # 小时分布
        hourly_analysis = behavior_data['hourly_analysis']
        axes[1, 0].plot(hourly_analysis.index, hourly_analysis.values, marker='o')
        axes[1, 0].set_title('用户行为时间分布')
        axes[1, 0].set_xlabel('小时')
        axes[1, 0].set_ylabel('行为数量')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 用户等级行为热力图
        user_level_analysis = behavior_data['user_level_analysis']
        sns.heatmap(user_level_analysis, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1, 1])
        axes[1, 1].set_title('用户等级行为热力图')
        axes[1, 1].set_xlabel('行为类型')
        axes[1, 1].set_ylabel('用户等级')
        
        plt.tight_layout()
        return fig
    
    def plot_cohort_heatmap(self, cohort_table):
        """绘制用户留存热力图"""
        plt.figure(figsize=(15, 8))
        
        # 只显示前12个月的数据
        cohort_display = cohort_table.iloc[:, :12] if cohort_table.shape[1] > 12 else cohort_table
        
        sns.heatmap(cohort_display, 
                   annot=True, 
                   fmt='.2%', 
                   cmap='YlOrRd',
                   linewidths=0.5)
        
        plt.title('用户留存分析热力图', fontsize=16, fontweight='bold')
        plt.xlabel('注册后月数')
        plt.ylabel('注册月份')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_ab_test_results(self, ab_results):
        """绘制A/B测试结果"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('A/B测试结果分析', fontsize=16, fontweight='bold')
        
        # 组间对比
        groups = ['Bronze用户', 'Gold用户']
        means = [ab_results['group_a_mean'], ab_results['group_b_mean']]
        
        bars = axes[0].bar(groups, means, color=['lightblue', 'gold'])
        axes[0].set_title('组间平均值对比')
        axes[0].set_ylabel('平均订单金额（元）')
        
        # 添加数值标签
        for bar, mean in zip(bars, means):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{mean:.2f}', ha='center', va='bottom')
        
        # 效果量和置信区间
        ci_lower = ab_results['ci_lower']
        ci_upper = ab_results['ci_upper']
        difference = ab_results['difference']
        
        axes[1].errorbar([0], [difference], yerr=[[difference-ci_lower], [ci_upper-difference]], 
                        fmt='o', capsize=10, capthick=2, markersize=8)
        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1].set_xlim(-0.5, 0.5)
        axes[1].set_xticks([0])
        axes[1].set_xticklabels(['效果差异'])
        axes[1].set_ylabel('金额差异（元）')
        axes[1].set_title(f'效果差异及95%置信区间\n(p-value: {ab_results["p_value"]:.4f})')
        axes[1].grid(True, alpha=0.3)
        
        # 添加显著性标注
        if ab_results['significant']:
            axes[1].text(0, difference, '显著', ha='center', va='bottom', 
                        fontweight='bold', color='green')
        else:
            axes[1].text(0, difference, '不显著', ha='center', va='bottom', 
                        fontweight='bold', color='red')
        
        plt.tight_layout()
        return fig
    
    def create_dashboard_summary(self, stats_data, funnel_data, rfm_data):
        """创建仪表板摘要"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('基础指标', '转化漏斗', 'RFM用户分群', '关键洞察'),
            specs=[[{"type": "indicator"}, {"type": "funnel"}],
                   [{"type": "pie"}, {"type": "table"}]]
        )
        
        # 关键指标
        conversion_rate = (stats_data['购买用户数'] / stats_data['活跃用户数']) * 100
        fig.add_trace(
            go.Indicator(
                mode="number+gauge+delta",
                value=conversion_rate,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "转化率 (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 25], 'color': "lightgray"},
                                {'range': [25, 50], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            ),
            row=1, col=1
        )
        
        # 转化漏斗
        fig.add_trace(
            go.Funnel(
                y=funnel_data['step_name'],
                x=funnel_data['user_count'],
                textinfo="value+percent initial"
            ),
            row=1, col=2
        )
        
        # RFM用户分群
        segment_counts = rfm_data['customer_segment'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=segment_counts.index,
                values=segment_counts.values,
                name="用户分群"
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=800, showlegend=False, title_text="电商数据分析仪表板")
        
        return fig

# 示例使用
if __name__ == "__main__":
    # 这里可以添加测试代码
    print("可视化模块已就绪！")
    visualizer = EcommerceVisualizer()
    print("可视化器已初始化") 