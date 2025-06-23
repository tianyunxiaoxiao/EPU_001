"""
EPU与波动率数据合并模块
将EPU数据与市场波动率数据合并为最终的分析数据集
"""
import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config

class EPUVolatilityMerger:
    """EPU和波动率数据合并器"""
    
    def __init__(self):
        self.volatility_file = Config.VOLATILITY_FILE
        self.volatility_columns = Config.VOLATILITY_COLUMNS
        self.epu_type_names = Config.EPU_TYPE_NAMES
    
    def load_volatility_data(self):
        """加载波动率数据"""
        try:
            logging.info(f"加载波动率数据: {self.volatility_file}")
            
            # 读取Excel文件
            df = pd.read_excel(self.volatility_file)
            
            # 确保DateTime列是日期格式
            if 'DateTime' in df.columns:
                df['DateTime'] = pd.to_datetime(df['DateTime'])
                df.rename(columns={'DateTime': 'Date'}, inplace=True)
            
            # 过滤需要的列
            available_columns = ['Date'] + [col for col in self.volatility_columns if col in df.columns]
            df = df[available_columns]
            
            # 按日期排序
            df = df.sort_values('Date').reset_index(drop=True)
            
            logging.info(f"加载波动率数据完成: {len(df)} 行, 时间范围: {df['Date'].min()} 到 {df['Date'].max()}")
            logging.info(f"可用列: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logging.error(f"加载波动率数据失败: {str(e)}")
            return None
    
    def load_epu_data(self, epu_file):
        """加载EPU数据"""
        try:
            logging.info(f"加载EPU数据: {epu_file}")
            
            if epu_file.endswith('.csv'):
                df = pd.read_csv(epu_file)
            elif epu_file.endswith('.json'):
                df = pd.read_json(epu_file)
            else:
                raise ValueError(f"不支持的文件格式: {epu_file}")
            
            # 确保Date列是日期格式
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            
            # 按日期排序
            df = df.sort_values('Date').reset_index(drop=True)
            
            logging.info(f"加载EPU数据完成: {len(df)} 行, 时间范围: {df['Date'].min()} 到 {df['Date'].max()}")
            
            return df
            
        except Exception as e:
            logging.error(f"加载EPU数据失败: {str(e)}")
            return None
    
    def merge_data(self, epu_df, volatility_df):
        """合并EPU和波动率数据"""
        try:
            logging.info("开始合并EPU和波动率数据...")
            
            # 使用Date列进行合并
            merged_df = pd.merge(
                epu_df, 
                volatility_df, 
                on='Date', 
                how='inner',  # 使用内连接，只保留两个数据集都有的日期
                suffixes=('_epu', '_vol')
            )
            
            # 重新排序列
            column_order = ['Date']
            
            # 添加EPU列
            epu_columns = []
            for epu_type in Config.EPU_TYPES:
                epu_name = Config.EPU_TYPE_NAMES[epu_type]
                if epu_name in merged_df.columns:
                    epu_columns.append(epu_name)
            
            column_order.extend(epu_columns)
            
            # 添加波动率列
            vol_columns = [col for col in self.volatility_columns if col in merged_df.columns]
            column_order.extend(vol_columns)
            
            # 添加其他列
            other_columns = [col for col in merged_df.columns if col not in column_order]
            column_order.extend(other_columns)
            
            # 重新排序
            available_columns = [col for col in column_order if col in merged_df.columns]
            merged_df = merged_df[available_columns]
            
            logging.info(f"数据合并完成: {len(merged_df)} 行")
            logging.info(f"时间范围: {merged_df['Date'].min()} 到 {merged_df['Date'].max()}")
            
            return merged_df
            
        except Exception as e:
            logging.error(f"数据合并失败: {str(e)}")
            return None
    
    def add_derived_features(self, df):
        """添加衍生特征"""
        try:
            logging.info("添加衍生特征...")
            
            # 计算EPU相关的衍生特征
            epu_columns = []
            for epu_type in Config.EPU_TYPES:
                epu_name = Config.EPU_TYPE_NAMES[epu_type]
                if epu_name in df.columns:
                    epu_columns.append(epu_name)
            
            if len(epu_columns) >= 2:
                # EPU总分
                df['EPU_Total'] = df[epu_columns].sum(axis=1)
                
                # EPU平均分
                df['EPU_Average'] = df[epu_columns].mean(axis=1)
                
                # EPU变异系数
                df['EPU_CV'] = df[epu_columns].std(axis=1) / (df[epu_columns].mean(axis=1) + 1e-8)
                
                # EPU最大最小比
                df['EPU_MaxMin_Ratio'] = df[epu_columns].max(axis=1) / (df[epu_columns].min(axis=1) + 1e-8)
            
            # 添加时间特征
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Quarter'] = df['Date'].dt.quarter
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            df['DayOfYear'] = df['Date'].dt.dayofyear
            
            # 计算滞后特征
            for epu_col in epu_columns:
                if epu_col in df.columns:
                    # 1天滞后
                    df[f'{epu_col}_Lag1'] = df[epu_col].shift(1)
                    # 7天滞后
                    df[f'{epu_col}_Lag7'] = df[epu_col].shift(7)
                    # 30天滞后
                    df[f'{epu_col}_Lag30'] = df[epu_col].shift(30)
            
            # 计算移动平均
            for epu_col in epu_columns:
                if epu_col in df.columns:
                    # 7天移动平均
                    df[f'{epu_col}_MA7'] = df[epu_col].rolling(window=7, min_periods=1).mean()
                    # 30天移动平均
                    df[f'{epu_col}_MA30'] = df[epu_col].rolling(window=30, min_periods=1).mean()
            
            # 计算波动率相关特征
            vol_columns = [col for col in self.volatility_columns if col in df.columns]
            
            if len(vol_columns) >= 2:
                # 波动率总和
                df['Volatility_Total'] = df[vol_columns].sum(axis=1)
                
                # 波动率平均
                df['Volatility_Average'] = df[vol_columns].mean(axis=1)
            
            logging.info(f"衍生特征添加完成，当前列数: {len(df.columns)}")
            
            return df
            
        except Exception as e:
            logging.error(f"添加衍生特征失败: {str(e)}")
            return df
    
    def clean_merged_data(self, df):
        """清理合并后的数据"""
        try:
            logging.info("清理合并数据...")
            
            original_rows = len(df)
            
            # 删除重复行
            df = df.drop_duplicates(subset=['Date']).reset_index(drop=True)
            
            # 处理缺失值
            # 对于EPU列，用0填充
            epu_columns = []
            for epu_type in Config.EPU_TYPES:
                epu_name = Config.EPU_TYPE_NAMES[epu_type]
                if epu_name in df.columns:
                    epu_columns.append(epu_name)
                    df[epu_name] = df[epu_name].fillna(0)
            
            # 对于波动率列，使用前向填充
            vol_columns = [col for col in self.volatility_columns if col in df.columns]
            for col in vol_columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            
            # 删除仍有缺失值的行
            before_drop = len(df)
            df = df.dropna(subset=epu_columns + vol_columns)
            after_drop = len(df)
            
            if before_drop > after_drop:
                logging.info(f"删除了 {before_drop - after_drop} 行缺失值数据")
            
            logging.info(f"数据清理完成: {original_rows} -> {len(df)} 行")
            
            return df
            
        except Exception as e:
            logging.error(f"数据清理失败: {str(e)}")
            return df
    
    def save_merged_data(self, df, output_file):
        """保存合并后的数据"""
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 保存为CSV
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logging.info(f"保存合并数据: {output_file}")
            
            # 同时保存为JSON
            json_file = output_file.replace('.csv', '.json')
            df.to_json(json_file, orient='records', date_format='iso', indent=2, force_ascii=False)
            logging.info(f"保存JSON格式: {json_file}")
            
            # 保存数据摘要
            summary_file = output_file.replace('.csv', '_summary.txt')
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("=== 合并数据摘要 ===\n")
                f.write(f"总行数: {len(df)}\n")
                f.write(f"总列数: {len(df.columns)}\n")
                f.write(f"时间范围: {df['Date'].min()} 到 {df['Date'].max()}\n\n")
                
                f.write("列信息:\n")
                for col in df.columns:
                    f.write(f"  {col}: {df[col].dtype}\n")
                
                f.write("\n数据统计:\n")
                f.write(str(df.describe()))
            
            return True
            
        except Exception as e:
            logging.error(f"保存合并数据失败: {str(e)}")
            return False
    
    def process_merge(self, epu_file, output_file=None):
        """处理完整的合并流程"""
        # 加载数据
        volatility_df = self.load_volatility_data()
        if volatility_df is None:
            return None
        
        epu_df = self.load_epu_data(epu_file)
        if epu_df is None:
            return None
        
        # 合并数据
        merged_df = self.merge_data(epu_df, volatility_df)
        if merged_df is None:
            return None
        
        # 添加衍生特征
        merged_df = self.add_derived_features(merged_df)
        
        # 清理数据
        merged_df = self.clean_merged_data(merged_df)
        
        # 保存结果
        if output_file:
            success = self.save_merged_data(merged_df, output_file)
            if not success:
                return None
        
        return merged_df
    
    def get_data_summary(self, df):
        """获取数据摘要"""
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'date_range': {
                'start': df['Date'].min(),
                'end': df['Date'].max()
            },
            'epu_columns': [],
            'volatility_columns': [],
            'missing_data': {}
        }
        
        # EPU列统计
        for epu_type in Config.EPU_TYPES:
            epu_name = Config.EPU_TYPE_NAMES[epu_type]
            if epu_name in df.columns:
                summary['epu_columns'].append({
                    'name': epu_name,
                    'mean': df[epu_name].mean(),
                    'std': df[epu_name].std(),
                    'min': df[epu_name].min(),
                    'max': df[epu_name].max()
                })
        
        # 波动率列统计
        for col in self.volatility_columns:
            if col in df.columns:
                summary['volatility_columns'].append({
                    'name': col,
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                })
        
        # 缺失值统计
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                summary['missing_data'][col] = missing_count
        
        return summary

def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 确保目录存在
    Config.ensure_directories()
    
    # 创建合并器
    merger = EPUVolatilityMerger()
    
    # 查找最新的EPU日度数据文件
    epu_files = []
    if os.path.exists(Config.DAILY_EPU_TABLE_DIR):
        for file in os.listdir(Config.DAILY_EPU_TABLE_DIR):
            if file.endswith('.csv') and 'epu_daily' in file:
                epu_files.append(os.path.join(Config.DAILY_EPU_TABLE_DIR, file))
    
    if not epu_files:
        logging.error("没有找到EPU日度数据文件")
        return
    
    # 使用最新的文件
    epu_files.sort()
    latest_epu_file = epu_files[-1]
    
    logging.info(f"使用EPU文件: {latest_epu_file}")
    
    # 设置输出文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(Config.FINAL_TABLE_DIR, f"epu_volatility_merged_{timestamp}.csv")
    
    # 执行合并
    merged_df = merger.process_merge(latest_epu_file, output_file)
    
    if merged_df is not None:
        # 获取摘要信息
        summary = merger.get_data_summary(merged_df)
        
        print("\n=== 合并结果摘要 ===")
        print(f"总行数: {summary['total_rows']}")
        print(f"总列数: {summary['total_columns']}")
        print(f"时间范围: {summary['date_range']['start']} 到 {summary['date_range']['end']}")
        
        print(f"\nEPU列统计:")
        for epu_stat in summary['epu_columns']:
            print(f"  {epu_stat['name']}: 均值={epu_stat['mean']:.2f}, 标准差={epu_stat['std']:.2f}")
        
        print(f"\n波动率列统计:")
        for vol_stat in summary['volatility_columns']:
            print(f"  {vol_stat['name']}: 均值={vol_stat['mean']:.6f}, 标准差={vol_stat['std']:.6f}")
        
        if summary['missing_data']:
            print(f"\n缺失值统计:")
            for col, missing_count in summary['missing_data'].items():
                print(f"  {col}: {missing_count} 个缺失值")
        
        logging.info("EPU-波动率数据合并完成")
    else:
        logging.error("EPU-波动率数据合并失败")

if __name__ == "__main__":
    main()
