"""
EPU数据聚合模块
将原始EPU结果聚合为日度和月度数据
"""
import pandas as pd
import numpy as np
import json
import os
import sys
import logging
from datetime import datetime, timedelta
from collections import defaultdict

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config

class EPUAggregator:
    """EPU数据聚合器"""
    
    def __init__(self):
        self.epu_types = Config.EPU_TYPES
        self.epu_type_names = Config.EPU_TYPE_NAMES
    
    def load_raw_epu_results(self, start_date, end_date, experiment_dir=None):
        """加载原始EPU结果数据"""
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        all_results = {}
        current_dt = start_dt
        
        # 确定EPU结果文件目录
        if experiment_dir:
            # 从实验目录加载
            epu_results_dir = os.path.join(experiment_dir, "epu_raw_outputs")
        else:
            # 从默认目录加载
            epu_results_dir = Config.RAW_EPU_RESULTS_DIR
        
        while current_dt <= end_dt:
            date_str = current_dt.strftime('%Y-%m-%d')
            result_file = os.path.join(epu_results_dir, f"epu_results_{date_str}.json")
            
            if os.path.exists(result_file):
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        daily_results = json.load(f)
                        all_results[date_str] = daily_results
                        logging.debug(f"成功加载EPU结果: {result_file}")
                except Exception as e:
                    logging.error(f"读取EPU结果失败 {result_file}: {str(e)}")
            else:
                logging.debug(f"EPU结果文件不存在: {result_file}")
            
            current_dt += timedelta(days=1)
        
        logging.info(f"加载了 {len(all_results)} 天的EPU结果数据")
        return all_results
    
    def create_daily_epu_table(self, raw_results):
        """创建日度EPU表格"""
        daily_data = []
        
        for date_str, date_results in raw_results.items():
            row_data = {'Date': date_str}
            
            # 添加每种EPU类型的分数
            for epu_type in self.epu_types:
                if epu_type in date_results:
                    row_data[self.epu_type_names[epu_type]] = date_results[epu_type]['score']
                    row_data[f'{epu_type}_success'] = date_results[epu_type]['success']
                    row_data[f'{epu_type}_content_count'] = date_results[epu_type]['content_count']
                else:
                    row_data[self.epu_type_names[epu_type]] = 0
                    row_data[f'{epu_type}_success'] = False
                    row_data[f'{epu_type}_content_count'] = 0
            
            # 计算统计指标
            epu_scores = [row_data.get(self.epu_type_names[epu_type], 0) for epu_type in self.epu_types]
            valid_scores = [score for score in epu_scores if score > 0]
            
            row_data['EPU_Mean'] = np.mean(valid_scores) if valid_scores else 0
            row_data['EPU_Std'] = np.std(valid_scores) if len(valid_scores) > 1 else 0
            row_data['EPU_Max'] = max(valid_scores) if valid_scores else 0
            row_data['EPU_Min'] = min(valid_scores) if valid_scores else 0
            row_data['Valid_EPU_Count'] = len(valid_scores)
            
            daily_data.append(row_data)
        
        # 创建DataFrame
        df = pd.DataFrame(daily_data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df
    
    def create_monthly_summary(self, daily_df):
        """创建月度EPU汇总"""
        # 添加年月列
        daily_df['YearMonth'] = daily_df['Date'].dt.to_period('M')
        
        # 定义聚合函数
        agg_funcs = {}
        
        # EPU分数列使用平均值
        for epu_type in self.epu_types:
            epu_name = self.epu_type_names[epu_type]
            agg_funcs[epu_name] = 'mean'
        
        # 统计指标
        agg_funcs.update({
            'EPU_Mean': 'mean',
            'EPU_Std': 'mean',
            'EPU_Max': 'max',
            'EPU_Min': 'min',
            'Valid_EPU_Count': 'sum'
        })
        
        # 按月聚合
        monthly_df = daily_df.groupby('YearMonth').agg(agg_funcs).reset_index()
        
        # 计算每月的天数和有效天数
        monthly_df['Days_in_Month'] = monthly_df['YearMonth'].apply(
            lambda x: len(daily_df[daily_df['YearMonth'] == x])
        )
        
        # 重新计算月度统计指标
        for idx, row in monthly_df.iterrows():
            period = row['YearMonth']
            month_data = daily_df[daily_df['YearMonth'] == period]
            
            # 计算月度EPU波动率
            epu_scores = []
            for epu_type in self.epu_types:
                epu_name = self.epu_type_names[epu_type]
                scores = month_data[epu_name].values
                valid_scores = scores[scores > 0]
                epu_scores.extend(valid_scores)
            
            if len(epu_scores) > 1:
                monthly_df.at[idx, 'EPU_Volatility'] = np.std(epu_scores)
                monthly_df.at[idx, 'EPU_Range'] = max(epu_scores) - min(epu_scores)
            else:
                monthly_df.at[idx, 'EPU_Volatility'] = 0
                monthly_df.at[idx, 'EPU_Range'] = 0
        
        return monthly_df
    
    def save_daily_table(self, daily_df, output_file):
        """保存日度EPU表格"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        daily_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logging.info(f"保存日度EPU表格: {output_file}")
        
        # 跳过JSON保存以避免内存问题
        logging.info(f"日度EPU表格保存完成: {output_file}")
    
    def save_monthly_summary(self, monthly_df, output_file):
        """保存月度EPU汇总"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 转换YearMonth为字符串
        monthly_df['YearMonth'] = monthly_df['YearMonth'].astype(str)
        
        monthly_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logging.info(f"保存月度EPU汇总: {output_file}")
        
        # 跳过JSON保存以避免内存问题
        logging.info(f"月度EPU汇总保存完成: {output_file}")
    
    def process_epu_aggregation(self, start_date, end_date, experiment_dir=None):
        """处理EPU数据聚合"""
        # 加载原始EPU结果
        logging.info("加载原始EPU结果数据...")
        raw_results = self.load_raw_epu_results(start_date, end_date, experiment_dir)
        
        if not raw_results:
            logging.error("没有找到有效的EPU结果数据")
            return None, None
        
        # 创建日度EPU表格
        logging.info("创建日度EPU表格...")
        daily_df = self.create_daily_epu_table(raw_results)
        
        # 创建月度汇总
        logging.info("创建月度EPU汇总...")
        monthly_df = self.create_monthly_summary(daily_df)
        
        # 确定输出目录
        if experiment_dir:
            daily_output = os.path.join(experiment_dir, "epu_daily_merged.csv")
            monthly_output = os.path.join(experiment_dir, "epu_monthly_summary.csv")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            daily_output = os.path.join(Config.DAILY_EPU_TABLE_DIR, f"epu_daily_{timestamp}.csv")
            monthly_output = os.path.join(Config.MONTHLY_EPU_SUMMARY_DIR, f"epu_monthly_{timestamp}.csv")
        
        # 保存结果
        self.save_daily_table(daily_df, daily_output)
        self.save_monthly_summary(monthly_df, monthly_output)
        
        # 输出统计信息
        logging.info(f"EPU聚合完成:")
        logging.info(f"  日度数据: {len(daily_df)} 行, 时间范围: {daily_df['Date'].min()} 到 {daily_df['Date'].max()}")
        logging.info(f"  月度数据: {len(monthly_df)} 行")
        
        # 显示EPU分数统计
        for epu_type in self.epu_types:
            epu_name = self.epu_type_names[epu_type]
            if epu_name in daily_df.columns:
                mean_score = daily_df[epu_name].mean()
                std_score = daily_df[epu_name].std()
                logging.info(f"  {epu_name}: 均值={mean_score:.2f}, 标准差={std_score:.2f}")
        
        return daily_df, monthly_df
    
    def get_epu_statistics(self, daily_df):
        """获取EPU统计信息"""
        stats = {}
        
        for epu_type in self.epu_types:
            epu_name = self.epu_type_names[epu_type]
            if epu_name in daily_df.columns:
                scores = daily_df[epu_name]
                valid_scores = scores[scores > 0]
                
                stats[epu_type] = {
                    'name': epu_name,
                    'count': len(valid_scores),
                    'mean': valid_scores.mean() if len(valid_scores) > 0 else 0,
                    'std': valid_scores.std() if len(valid_scores) > 0 else 0,
                    'min': valid_scores.min() if len(valid_scores) > 0 else 0,
                    'max': valid_scores.max() if len(valid_scores) > 0 else 0,
                    'median': valid_scores.median() if len(valid_scores) > 0 else 0
                }
        
        return stats
    
    def create_epu_correlation_matrix(self, daily_df):
        """创建EPU类型间的相关性矩阵"""
        epu_columns = [self.epu_type_names[epu_type] for epu_type in self.epu_types]
        existing_columns = [col for col in epu_columns if col in daily_df.columns]
        
        if len(existing_columns) < 2:
            return None
        
        correlation_matrix = daily_df[existing_columns].corr()
        return correlation_matrix

def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 确保目录存在
    Config.ensure_directories()
    
    # 创建EPU聚合器
    aggregator = EPUAggregator()
    
    # 处理EPU聚合
    start_date = Config.START_DATE
    end_date = Config.END_DATE
    
    logging.info(f"开始EPU数据聚合: {start_date} 到 {end_date}")
    
    daily_df, monthly_df = aggregator.process_epu_aggregation(start_date, end_date)
    
    if daily_df is not None:
        # 获取统计信息
        stats = aggregator.get_epu_statistics(daily_df)
        print("\n=== EPU统计信息 ===")
        for epu_type, stat in stats.items():
            print(f"{stat['name']}: 均值={stat['mean']:.2f}, 标准差={stat['std']:.2f}, 样本数={stat['count']}")
        
        # 计算相关性矩阵
        corr_matrix = aggregator.create_epu_correlation_matrix(daily_df)
        if corr_matrix is not None:
            print("\n=== EPU相关性矩阵 ===")
            print(corr_matrix.round(3))
    
    logging.info("EPU数据聚合完成")

if __name__ == "__main__":
    main()
