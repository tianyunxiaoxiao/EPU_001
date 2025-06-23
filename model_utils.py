"""
建模工具函数模块
提供建模过程中的通用工具函数
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import os
import sys
import logging
from datetime import datetime
import pickle
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config

class ModelUtils:
    """建模工具类"""
    
    def __init__(self):
        self.setup_chinese_font()
        self.scalers = {}
        
    def setup_chinese_font(self):
        """设置中文字体"""
        try:
            font_path = Config.FONT_PATH
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            logging.warning(f"中文字体设置失败: {str(e)}")
    
    def load_merged_data(self, data_file):
        """加载合并后的数据"""
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
            
            logging.info(f"加载数据完成: {len(df)} 行, {len(df.columns)} 列")
            return df
            
        except Exception as e:
            logging.error(f"加载数据失败: {str(e)}")
            return None
    
    def prepare_features_targets(self, df):
        """准备特征和目标变量"""
        # EPU特征列
        epu_features = []
        for epu_type in Config.EPU_TYPES:
            epu_name = Config.EPU_TYPE_NAMES[epu_type]
            if epu_name in df.columns:
                epu_features.append(epu_name)
        
        # 波动率目标列
        volatility_targets = [col for col in Config.VOLATILITY_COLUMNS if col in df.columns]
        
        # 其他特征列（时间特征、滞后特征等）
        other_features = []
        for col in df.columns:
            if any([
                col.endswith('_Lag1'),
                col.endswith('_Lag7'), 
                col.endswith('_Lag30'),
                col.endswith('_MA7'),
                col.endswith('_MA30'),
                col in ['Year', 'Month', 'Quarter', 'DayOfWeek', 'DayOfYear'],
                col.startswith('EPU_')
            ]):
                other_features.append(col)
        
        all_features = epu_features + other_features
        available_features = [col for col in all_features if col in df.columns]
        
        return available_features, volatility_targets
    
    def create_feature_target_pairs(self, df, features, targets):
        """创建特征-目标变量配对"""
        pairs = []
        
        # 单个EPU特征对单个波动率目标
        epu_features = [f for f in features if f in Config.EPU_TYPE_NAMES.values()]
        
        for epu_feature in epu_features:
            for target in targets:
                pairs.append({
                    'features': [epu_feature],
                    'target': target,
                    'name': f"{epu_feature}_to_{target}",
                    'type': 'single_epu'
                })
        
        # 所有EPU特征对单个波动率目标
        if len(epu_features) > 1:
            for target in targets:
                pairs.append({
                    'features': epu_features,
                    'target': target,
                    'name': f"all_epu_to_{target}",
                    'type': 'multi_epu'
                })
        
        # 所有特征对单个波动率目标
        for target in targets:
            pairs.append({
                'features': features,
                'target': target,
                'name': f"all_features_to_{target}",
                'type': 'full_features'
            })
        
        return pairs
    
    def split_data(self, df, features, target, test_size=0.2, time_split=True):
        """分割数据集"""
        # 准备数据
        X = df[features].copy()
        y = df[target].copy()
        
        # 删除缺失值
        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_idx]
        y = y[valid_idx]
        if 'Date' in df.columns:
            dates = df['Date'][valid_idx]
        elif df.index.name == 'Date' or hasattr(df.index, 'date'):
            dates = df.index[valid_idx]
        else:
            dates = None
        
        if len(X) == 0:
            raise ValueError("没有有效的训练数据")
        
        if time_split and dates is not None:
            # 基于时间分割
            split_date = dates.quantile(1 - test_size)
            train_mask = dates <= split_date
            test_mask = dates > split_date
            
            X_train = X[train_mask]
            X_test = X[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]
            
        else:
            # 随机分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, 
                random_state=Config.MODEL_CONFIGS['random_state']
            )
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test, method='standard'):
        """特征标准化"""
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            return X_train, X_test, None
        
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        return X_train_scaled, X_test_scaled, scaler
    
    def evaluate_model(self, y_true, y_pred, model_name="Model"):
        """评估模型性能"""
        metrics = {
            'model_name': model_name,
            'r2_score': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'n_samples': len(y_true)
        }
        
        # 添加相对指标
        y_mean = np.mean(y_true)
        if y_mean != 0:
            metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics['rmse_relative'] = metrics['rmse'] / y_mean * 100
        
        return metrics
    
    def cross_validate_model(self, model, X, y, cv_folds=5, scoring='r2'):
        """交叉验证模型"""
        # 使用时间序列分割
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring)
        
        return {
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'cv_folds': cv_folds
        }
    
    def plot_predictions(self, y_true, y_pred, title="Model Predictions", save_path=None):
        """绘制预测结果对比图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 散点图
        ax1.scatter(y_true, y_pred, alpha=0.6)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax1.set_xlabel('实际值')
        ax1.set_ylabel('预测值')
        ax1.set_title(f'{title} - 预测vs实际')
        ax1.grid(True, alpha=0.3)
        
        # 残差图
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('预测值')
        ax2.set_ylabel('残差')
        ax2.set_title(f'{title} - 残差分析')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"预测图保存至: {save_path}")
        
        return fig
    
    def plot_feature_importance(self, importance_scores, feature_names, title="Feature Importance", save_path=None):
        """绘制特征重要性图"""
        if len(importance_scores) != len(feature_names):
            logging.error("特征重要性数量与特征名称数量不匹配")
            return None
        
        # 创建DataFrame并排序
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=True)
        
        # 绘图
        plt.figure(figsize=(10, max(6, len(feature_names) * 0.4)))
        bars = plt.barh(importance_df['feature'], importance_df['importance'])
        
        # 设置颜色
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.xlabel('重要性')
        plt.title(title)
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"特征重要性图保存至: {save_path}")
        
        return plt.gcf()
    
    def save_model(self, model, model_info, save_path):
        """保存模型"""
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存模型
            model_file = save_path.replace('.json', '.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            
            # 保存模型信息
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, ensure_ascii=False, indent=2, default=str)
            
            logging.info(f"模型保存至: {model_file}")
            logging.info(f"模型信息保存至: {save_path}")
            
            return True
            
        except Exception as e:
            logging.error(f"模型保存失败: {str(e)}")
            return False
    
    def load_model(self, model_file):
        """加载模型"""
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            info_file = model_file.replace('.pkl', '.json')
            if os.path.exists(info_file):
                with open(info_file, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
            else:
                model_info = {}
            
            return model, model_info
            
        except Exception as e:
            logging.error(f"模型加载失败: {str(e)}")
            return None, None
    
    def create_model_summary(self, results_list):
        """创建模型结果汇总"""
        summary_df = pd.DataFrame(results_list)
        
        # 按R²分数排序
        summary_df = summary_df.sort_values('r2_score', ascending=False).reset_index(drop=True)
        
        # 添加排名
        summary_df['rank'] = range(1, len(summary_df) + 1)
        
        # 重新排列列顺序
        columns_order = ['rank', 'model_name', 'feature_target_pair', 'r2_score', 'rmse', 'mae']
        if 'cv_mean' in summary_df.columns:
            columns_order.append('cv_mean')
        
        existing_columns = [col for col in columns_order if col in summary_df.columns]
        other_columns = [col for col in summary_df.columns if col not in existing_columns]
        final_columns = existing_columns + other_columns
        
        summary_df = summary_df[final_columns]
        
        return summary_df
    
    def plot_model_comparison(self, summary_df, metric='r2_score', save_path=None):
        """绘制模型比较图"""
        plt.figure(figsize=(12, 8))
        
        # 选择前20个模型
        top_models = summary_df.head(20)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_models)))
        bars = plt.barh(range(len(top_models)), top_models[metric], color=colors)
        
        plt.yticks(range(len(top_models)), top_models['model_name'])
        plt.xlabel(metric.upper())
        plt.title(f'模型性能比较 ({metric.upper()})')
        plt.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, top_models[metric])):
            plt.text(value + 0.001, i, f'{value:.3f}', 
                    va='center', ha='left', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"模型比较图保存至: {save_path}")
        
        return plt.gcf()

def get_latest_merged_data():
    """获取最新的合并数据文件"""
    final_table_dir = Config.FINAL_TABLE_DIR
    
    if not os.path.exists(final_table_dir):
        return None
    
    csv_files = [f for f in os.listdir(final_table_dir) if f.endswith('.csv')]
    
    if not csv_files:
        return None
    
    # 按修改时间排序，取最新的
    csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(final_table_dir, x)))
    latest_file = os.path.join(final_table_dir, csv_files[-1])
    
    return latest_file

def setup_experiment_dir(experiment_id=None):
    """设置实验目录"""
    if experiment_id is None:
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    experiment_dir = Config.get_experiment_dir(experiment_id)
    
    # 创建子目录
    subdirs = ['models', 'plots', 'results']
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
    
    return experiment_dir

def main():
    """测试函数"""
    utils = ModelUtils()
    
    # 测试数据加载
    latest_data_file = get_latest_merged_data()
    if latest_data_file:
        df = utils.load_merged_data(latest_data_file)
        if df is not None:
            print(f"测试数据加载成功: {len(df)} 行")
            
            # 测试特征目标准备
            features, targets = utils.prepare_features_targets(df)
            print(f"特征列数: {len(features)}")
            print(f"目标列数: {len(targets)}")
            
            # 测试特征目标配对
            pairs = utils.create_feature_target_pairs(df, features, targets)
            print(f"特征-目标配对数: {len(pairs)}")
    else:
        print("未找到合并数据文件")

if __name__ == "__main__":
    main()
