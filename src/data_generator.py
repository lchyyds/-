# -*- coding: utf-8 -*-
"""
电商数据生成器
生成模拟的用户、商品、行为和订单数据
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CONFIG, PATHS

class EcommerceDataGenerator:
    def __init__(self, config=DATA_CONFIG):
        self.config = config
        self.fake = Faker('zh_CN')  # 使用中文
        Faker.seed(config['random_seed'])
        np.random.seed(config['random_seed'])
        random.seed(config['random_seed'])
        
        # 创建输出目录
        os.makedirs(PATHS['raw_data'], exist_ok=True)
        
    def generate_users(self):
        """生成用户数据"""
        print("生成用户数据...")
        
        users = []
        for i in range(self.config['n_users']):
            user = {
                'user_id': f'U{i+1:06d}',
                'username': self.fake.user_name(),
                'age': np.random.randint(18, 65),
                'gender': np.random.choice(['M', 'F'], p=[0.52, 0.48]),
                'city': self.fake.city(),
                'register_date': self.fake.date_between(start_date='-2y', end_date='today'),
                'user_level': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], 
                                             p=[0.4, 0.3, 0.2, 0.1])
            }
            users.append(user)
            
        return pd.DataFrame(users)
    
    def generate_products(self):
        """生成商品数据"""
        print("生成商品数据...")
        
        categories = ['服装', '电子产品', '家居用品', '美妆护肤', '食品饮料', 
                     '运动户外', '图书文具', '母婴用品', '数码配件', '家电',
                     '珠宝饰品', '汽车用品', '宠物用品', '工具五金', '医疗保健',
                     '办公用品', '乐器', '玩具', '园艺用品', '其他']
        
        products = []
        for i in range(self.config['n_products']):
            category = np.random.choice(categories)
            base_price = np.random.exponential(100) + 10  # 指数分布价格
            
            product = {
                'product_id': f'P{i+1:05d}',
                'product_name': f'{category}商品{i+1}',
                'category': category,
                'price': round(base_price, 2),
                'brand': self.fake.company(),
                'rating': np.random.normal(4.2, 0.8),  # 评分分布
                'review_count': max(0, int(np.random.exponential(50))),
                'stock': np.random.randint(0, 1000)
            }
            products.append(product)
            
        return pd.DataFrame(products)
    
    def generate_user_behaviors(self, users_df, products_df):
        """生成用户行为数据"""
        print("生成用户行为数据...")
        
        behaviors = []
        start_date = datetime.now() - timedelta(days=self.config['date_range'])
        
        # 为每个用户生成行为
        for _, user in users_df.iterrows():
            # 根据用户等级调整活跃度
            activity_multiplier = {'Bronze': 1, 'Silver': 2, 'Gold': 3, 'Platinum': 5}
            n_behaviors = np.random.poisson(20 * activity_multiplier[user['user_level']])
            
            for _ in range(n_behaviors):
                # 随机选择商品
                product = products_df.sample(1).iloc[0]
                
                # 生成行为时间
                behavior_date = start_date + timedelta(
                    days=np.random.randint(0, self.config['date_range']),
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60)
                )
                
                # 行为类型的概率分布
                behavior_type = np.random.choice(
                    ['browse', 'cart', 'order', 'pay'], 
                    p=[0.7, 0.2, 0.08, 0.02]  # 浏览>加购>下单>支付
                )
                
                behavior = {
                    'user_id': user['user_id'],
                    'product_id': product['product_id'],
                    'behavior_type': behavior_type,
                    'behavior_time': behavior_date,
                    'session_id': f'S{np.random.randint(1, 100000):06d}',
                    'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], 
                                                  p=[0.6, 0.35, 0.05])
                }
                behaviors.append(behavior)
        
        return pd.DataFrame(behaviors)
    
    def generate_orders(self, behaviors_df, users_df, products_df):
        """基于行为数据生成订单数据"""
        print("生成订单数据...")
        
        # 筛选出下单和支付行为
        order_behaviors = behaviors_df[behaviors_df['behavior_type'].isin(['order', 'pay'])]
        
        orders = []
        for _, behavior in order_behaviors.iterrows():
            user = users_df[users_df['user_id'] == behavior['user_id']].iloc[0]
            product = products_df[products_df['product_id'] == behavior['product_id']].iloc[0]
            
            # 订单状态
            if behavior['behavior_type'] == 'pay':
                status = 'completed'
            else:
                status = np.random.choice(['pending', 'cancelled'], p=[0.7, 0.3])
            
            order = {
                'order_id': f'O{len(orders)+1:08d}',
                'user_id': behavior['user_id'],
                'product_id': behavior['product_id'],
                'quantity': np.random.randint(1, 5),
                'unit_price': product['price'],
                'total_amount': product['price'] * np.random.randint(1, 5),
                'order_time': behavior['behavior_time'],
                'status': status,
                'payment_method': np.random.choice(['credit_card', 'alipay', 'wechat_pay'], 
                                                 p=[0.3, 0.4, 0.3])
            }
            orders.append(order)
            
        return pd.DataFrame(orders)
    
    def save_data(self, users_df, products_df, behaviors_df, orders_df):
        """保存数据到文件"""
        print("保存数据文件...")
        
        users_df.to_csv(f"{PATHS['raw_data']}users.csv", index=False, encoding='utf-8')
        products_df.to_csv(f"{PATHS['raw_data']}products.csv", index=False, encoding='utf-8')
        behaviors_df.to_csv(f"{PATHS['raw_data']}user_behaviors.csv", index=False, encoding='utf-8')
        orders_df.to_csv(f"{PATHS['raw_data']}orders.csv", index=False, encoding='utf-8')
        
        print(f"数据已保存到 {PATHS['raw_data']} 目录")
        print(f"- 用户数据: {len(users_df)} 条")
        print(f"- 商品数据: {len(products_df)} 条")
        print(f"- 行为数据: {len(behaviors_df)} 条")
        print(f"- 订单数据: {len(orders_df)} 条")
    
    def generate_all_data(self):
        """生成所有数据"""
        print("开始生成电商数据...")
        
        # 生成各类数据
        users_df = self.generate_users()
        products_df = self.generate_products()
        behaviors_df = self.generate_user_behaviors(users_df, products_df)
        orders_df = self.generate_orders(behaviors_df, users_df, products_df)
        
        # 保存数据
        self.save_data(users_df, products_df, behaviors_df, orders_df)
        
        return users_df, products_df, behaviors_df, orders_df

if __name__ == "__main__":
    generator = EcommerceDataGenerator()
    generator.generate_all_data()
    print("数据生成完成！") 