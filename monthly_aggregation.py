"""
月度数据聚合模块
将日度EPU和波动率数据聚合为月度数据
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import sys
import logging
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config

class MonthlyAggregator:
    """月度数据聚合器"""
    
    def __init__(self):
        self.setup_chinese_font()
        
    def setup_chinese_font(self):
        """设置中文字体"""
        try:
            font_path = Config.FONT_PATH
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            logging.warning(f"中文字体设置失败: {str(e)}")
    
    def load_daily_data(self, data_file):
        """加载日度数据"""
        try:
            if data_file.endswith('.csv'):
                df = pd.read_csv(data_file)
            elif data_file.endswith('.json'):
                df = pd.read_json(data_file)
            else:
                raise ValueError(f"不支持的文件格式: {data_file}")
            
            # 确保Date列是日期格式
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date').reset_index(drop=True)
            
            logging.info(f"加载日度数据完成: {len(df)} 行, {len(df.columns)} 列")
            return df
            
        except Exception as e:
            logging.error(f"加载数据失败: {str(e)}")
            return None
    
    def identify_data_types(self, df):
        """识别数据类型"""
        epu_columns = []
        volatility_columns = []
        news_columns = []
        other_columns = []
        
        for col in df.columns:
            if col == 'Date':
                continue
            elif any(epu_name in col for epu_name in Config.EPU_TYPE_NAMES.values()):
                epu_columns.append(col)
            elif any(vol_col in col for vol_col in Config.VOLATILITY_COLUMNS):
                volatility_columns.append(col)
            elif '新闻' in col or 'news' in col.lower() or 'content' in col.lower():
                news_columns.append(col)
            else:
                other_columns.append(col)
        
        return {
            'epu_columns': epu_columns,
            'volatility_columns': volatility_columns,
            'news_columns': news_columns,
            'other_columns': other_columns
        }
    
    def aggregate_epu_data(self, df, epu_columns, method='mean'):
        """聚合EPU数据"""
        monthly_epu = {}
        
        for col in epu_columns:
            if col not in df.columns:
                continue
                
            if method == 'mean':
                monthly_epu[col] = df.groupby(df['Date'].dt.to_period('M'))[col].mean()
            elif method == 'median':
                monthly_epu[col] = df.groupby(df['Date'].dt.to_period('M'))[col].median()
            elif method == 'sum':
                monthly_epu[col] = df.groupby(df['Date'].dt.to_period('M'))[col].sum()
            elif method == 'last':
                monthly_epu[col] = df.groupby(df['Date'].dt.to_period('M'))[col].last()
            else:
                monthly_epu[col] = df.groupby(df['Date'].dt.to_period('M'))[col].mean()
        
        return pd.DataFrame(monthly_epu)
    
    def aggregate_volatility_data(self, df, volatility_columns):
        """聚合波动率数据"""
        monthly_vol = {}
        
        for col in volatility_columns:
            if col not in df.columns:
                continue
            
            monthly_data = df.groupby(df['Date'].dt.to_period('M'))[col].agg({
                f'{col}_mean': 'mean',
                f'{col}_std': 'std',
                f'{col}_min': 'min',
                f'{col}_max': 'max',
                f'{col}_last': 'last'
            })
            
            for new_col in monthly_data.columns:
                monthly_vol[new_col] = monthly_data[new_col]
        
        return pd.DataFrame(monthly_vol)
    
    def aggregate_news_data(self, df, news_columns):
        """聚合新闻数据"""
        monthly_news = {}
        
        for col in news_columns:
            if col not in df.columns:
                continue
            
            # 新闻条数统计
            monthly_news[f'{col}_count'] = df.groupby(df['Date'].dt.to_period('M'))[col].count()
            
            # 新闻内容长度统计
            if df[col].dtype == 'object':
                df[f'{col}_length'] = df[col].astype(str).str.len()
                monthly_news[f'{col}_avg_length'] = df.groupby(df['Date'].dt.to_period('M'))[f'{col}_length'].mean()
        
        return pd.DataFrame(monthly_news)
    
    def calculate_monthly_features(self, monthly_df):
        """计算月度特征"""
        # 添加时间特征
        monthly_df['Year'] = monthly_df.index.year
        monthly_df['Month'] = monthly_df.index.month
        monthly_df['Quarter'] = monthly_df.index.quarter
        
        # 计算滞后特征
        epu_cols = [col for col in monthly_df.columns if any(epu_name in col for epu_name in Config.EPU_TYPE_NAMES.values())]
        vol_cols = [col for col in monthly_df.columns if any(vol_col in col for vol_col in Config.VOLATILITY_COLUMNS)]
        
        # EPU滞后特征
        for col in epu_cols:
            monthly_df[f'{col}_Lag1'] = monthly_df[col].shift(1)
            monthly_df[f'{col}_Lag3'] = monthly_df[col].shift(3)
            monthly_df[f'{col}_Lag6'] = monthly_df[col].shift(6)
            
            # 移动平均
            monthly_df[f'{col}_MA3'] = monthly_df[col].rolling(window=3).mean()
            monthly_df[f'{col}_MA6'] = monthly_df[col].rolling(window=6).mean()
            monthly_df[f'{col}_MA12'] = monthly_df[col].rolling(window=12).mean()
        
        # 波动率滞后特征
        for col in vol_cols:
            monthly_df[f'{col}_Lag1'] = monthly_df[col].shift(1)
            monthly_df[f'{col}_Lag3'] = monthly_df[col].shift(3)
            
            # 移动平均
            if monthly_df[col].notna().sum() > 3:
                monthly_df[f'{col}_MA3'] = monthly_df[col].rolling(window=3).mean()
                monthly_df[f'{col}_MA6'] = monthly_df[col].rolling(window=6).mean()
        
        return monthly_df
    
    def create_monthly_dataset(self, df, epu_method='mean', save_path=None):
        """创建月度数据集"""
        logging.info("开始创建月度数据集...")
        
        # 识别数据类型
        data_types = self.identify_data_types(df)
        
        logging.info(f"发现EPU列: {len(data_types['epu_columns'])}")
        logging.info(f"发现波动率列: {len(data_types['volatility_columns'])}")
        logging.info(f"发现新闻列: {len(data_types['news_columns'])}")
        
        # 聚合不同类型的数据
        monthly_datasets = []
        
        # EPU数据
        if data_types['epu_columns']:
            monthly_epu = self.aggregate_epu_data(df, data_types['epu_columns'], epu_method)
            monthly_datasets.append(monthly_epu)
            logging.info(f"EPU月度聚合完成: {len(monthly_epu)} 行")
        
        # 波动率数据
        if data_types['volatility_columns']:
            monthly_vol = self.aggregate_volatility_data(df, data_types['volatility_columns'])
            monthly_datasets.append(monthly_vol)
            logging.info(f"波动率月度聚合完成: {len(monthly_vol)} 行")
        
        # 新闻数据
        if data_types['news_columns']:
            monthly_news = self.aggregate_news_data(df, data_types['news_columns'])
            monthly_datasets.append(monthly_news)
            logging.info(f"新闻月度聚合完成: {len(monthly_news)} 行")
        
        # 合并所有月度数据
        if monthly_datasets:
            monthly_df = pd.concat(monthly_datasets, axis=1)
        else:
            logging.error("没有找到可聚合的数据")
            return None
        
        # 重置索引，将Period转换为日期
        monthly_df.index = monthly_df.index.to_timestamp()
        monthly_df = monthly_df.reset_index()
        monthly_df.rename(columns={'index': 'Date'}, inplace=True)
        
        # 计算衍生特征
        monthly_df = self.calculate_monthly_features(monthly_df)
        
        # 删除缺失值过多的列
        missing_threshold = 0.8
        missing_ratios = monthly_df.isnull().sum() / len(monthly_df)
        columns_to_keep = missing_ratios[missing_ratios < missing_threshold].index.tolist()
        monthly_df = monthly_df[columns_to_keep]
        
        logging.info(f"月度数据集创建完成: {len(monthly_df)} 行, {len(monthly_df.columns)} 列")
        
        # 保存结果
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            monthly_df.to_csv(save_path, index=False, encoding='utf-8-sig')
            logging.info(f"月度数据集保存至: {save_path}")
        
        return monthly_df
    
    def analyze_monthly_trends(self, monthly_df, save_dir=None):
        """分析月度趋势"""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 识别EPU和波动率列
        epu_cols = [col for col in monthly_df.columns 
                   if any(epu_name in col for epu_name in Config.EPU_TYPE_NAMES.values()) 
                   and not any(suffix in col for suffix in ['_Lag', '_MA'])]
        
        vol_cols = [col for col in monthly_df.columns 
                   if any(vol_col in col for vol_col in Config.VOLATILITY_COLUMNS)
                   and 'mean' in col]
        
        # 绘制EPU趋势
        if epu_cols and len(monthly_df) > 1:
            self.plot_epu_trends(monthly_df, epu_cols, save_dir)
        
        # 绘制波动率趋势
        if vol_cols and len(monthly_df) > 1:
            self.plot_volatility_trends(monthly_df, vol_cols, save_dir)
        
        # 计算相关性分析
        if epu_cols and vol_cols:
            self.analyze_correlations(monthly_df, epu_cols, vol_cols, save_dir)
        
        # 创建月度统计汇总
        self.create_monthly_summary(monthly_df, save_dir)
    
    def plot_epu_trends(self, monthly_df, epu_cols, save_dir=None):
        """绘制EPU趋势图"""
        n_epu = len(epu_cols)
        n_cols = 2
        n_rows = (n_epu + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(epu_cols):
            ax = axes[i] if n_epu > 1 else axes
            
            ax.plot(monthly_df['Date'], monthly_df[col], linewidth=2, marker='o', markersize=4)
            ax.set_title(f'{col} 月度趋势')
            ax.set_xlabel('日期')
            ax.set_ylabel('EPU指数')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        # 隐藏多余的子图
        for i in range(n_epu, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_dir:
            trend_file = os.path.join(save_dir, 'monthly_epu_trends.png')
            plt.savefig(trend_file, dpi=300, bbox_inches='tight')
            logging.info(f"EPU趋势图保存至: {trend_file}")
        
        return fig
    
    def plot_volatility_trends(self, monthly_df, vol_cols, save_dir=None):
        """绘制波动率趋势图"""
        n_vol = len(vol_cols)
        n_cols = 2
        n_rows = (n_vol + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(vol_cols):
            ax = axes[i] if n_vol > 1 else axes
            
            ax.plot(monthly_df['Date'], monthly_df[col], linewidth=2, marker='s', markersize=4, color='red')
            ax.set_title(f'{col} 月度趋势')
            ax.set_xlabel('日期')
            ax.set_ylabel('波动率')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        # 隐藏多余的子图
        for i in range(n_vol, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_dir:
            trend_file = os.path.join(save_dir, 'monthly_volatility_trends.png')
            plt.savefig(trend_file, dpi=300, bbox_inches='tight')
            logging.info(f"波动率趋势图保存至: {trend_file}")
        
        return fig
    
    def analyze_correlations(self, monthly_df, epu_cols, vol_cols, save_dir=None):
        """分析相关性"""
        # 计算相关性矩阵
        analysis_cols = epu_cols + vol_cols
        corr_data = monthly_df[analysis_cols].select_dtypes(include=[np.number])
        corr_matrix = corr_data.corr()
        
        # 绘制相关性热力图
        fig, ax = plt.subplots(figsize=(12, 10))
        
        import matplotlib.colors as mcolors
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "correlation", ["red", "white", "blue"], N=256
        )
        
        im = ax.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
        
        # 设置坐标轴
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(corr_matrix.index)))
        ax.set_yticklabels(corr_matrix.index)
        
        # 添加数值标签
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                value = corr_matrix.iloc[i, j]
                if not pd.isna(value):
                    text_color = 'white' if abs(value) > 0.5 else 'black'
                    ax.text(j, i, f'{value:.2f}', ha="center", va="center", color=text_color)
        
        plt.colorbar(im, ax=ax, label='相关系数')
        ax.set_title('EPU与波动率月度相关性分析')
        plt.tight_layout()
        
        if save_dir:
            corr_file = os.path.join(save_dir, 'monthly_correlation_matrix.png')
            plt.savefig(corr_file, dpi=300, bbox_inches='tight')
            logging.info(f"相关性图保存至: {corr_file}")
            
            # 保存相关性数据
            corr_csv_file = os.path.join(save_dir, 'monthly_correlation_matrix.csv')
            corr_matrix.to_csv(corr_csv_file, encoding='utf-8-sig')
        
        return corr_matrix
    
    def create_monthly_summary(self, monthly_df, save_dir=None):
        """创建月度统计汇总"""
        # 基本统计信息
        numeric_cols = monthly_df.select_dtypes(include=[np.number]).columns
        summary_stats = monthly_df[numeric_cols].describe()
        
        # 缺失值统计
        missing_stats = pd.DataFrame({
            '缺失值数量': monthly_df.isnull().sum(),
            '缺失率': monthly_df.isnull().sum() / len(monthly_df)
        })
        
        # 数据范围统计
        date_stats = {
            '数据起始日期': monthly_df['Date'].min(),
            '数据结束日期': monthly_df['Date'].max(),
            '数据月份数': len(monthly_df),
            '数据列数': len(monthly_df.columns)
        }
        
        if save_dir:
            # 保存统计汇总
            summary_file = os.path.join(save_dir, 'monthly_summary_statistics.csv')
            summary_stats.to_csv(summary_file, encoding='utf-8-sig')
            
            missing_file = os.path.join(save_dir, 'monthly_missing_values.csv')
            missing_stats.to_csv(missing_file, encoding='utf-8-sig')
            
            # 保存数据信息
            info_file = os.path.join(save_dir, 'monthly_data_info.txt')
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write("月度数据集信息\n")
                f.write("=" * 50 + "\n")
                for key, value in date_stats.items():
                    f.write(f"{key}: {value}\n")
            
            logging.info(f"月度汇总统计保存至: {save_dir}")
        
        return {
            'summary_stats': summary_stats,
            'missing_stats': missing_stats,
            'date_stats': date_stats
        }

def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 查找最新的合并数据文件
    final_table_dir = Config.FINAL_TABLE_DIR
    
    if not os.path.exists(final_table_dir):
        logging.error("未找到最终数据目录")
        return
    
    csv_files = [f for f in os.listdir(final_table_dir) if f.endswith('.csv')]
    
    if not csv_files:
        logging.error("未找到CSV数据文件")
        return
    
    # 选择最新的文件
    csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(final_table_dir, x)))
    latest_file = os.path.join(final_table_dir, csv_files[-1])
    
    logging.info(f"使用数据文件: {latest_file}")
    
    # 创建聚合器
    aggregator = MonthlyAggregator()
    
    # 加载数据
    df = aggregator.load_daily_data(latest_file)
    if df is None:
        return
    
    # 创建月度数据集
    monthly_save_path = os.path.join(final_table_dir, f"monthly_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    monthly_df = aggregator.create_monthly_dataset(df, epu_method='mean', save_path=monthly_save_path)
    
    if monthly_df is None:
        return
    
    # 分析月度趋势
    analysis_dir = os.path.join(final_table_dir, f"monthly_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    aggregator.analyze_monthly_trends(monthly_df, analysis_dir)
    
    print(f"\n月度数据聚合完成")
    print(f"月度数据集: {monthly_save_path}")
    print(f"分析结果: {analysis_dir}")
    print(f"月度数据: {len(monthly_df)} 行, {len(monthly_df.columns)} 列")

if __name__ == "__main__":
    main()
