"""
基于LSTM波动率预测的交易策略回测模块
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
import logging
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config
from src.modeling.model_utils import ModelUtils, get_latest_merged_data
from src.backtest.backtest_utils import BacktestUtils

class LSTMVolatilityStrategy:
    """基于LSTM波动率预测的交易策略"""
    
    def __init__(self, experiment_dir=None):
        self.utils = ModelUtils()
        self.backtest_utils = BacktestUtils()
        self.experiment_dir = experiment_dir
        self.strategy_results = {}
        
    def load_lstm_model(self, model_path):
        """加载训练好的LSTM模型"""
        try:
            model = tf.keras.models.load_model(model_path)
            
            # 加载模型信息
            info_path = model_path.replace('.h5', '.json')
            if os.path.exists(info_path):
                import json
                with open(info_path, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
            else:
                model_info = {}
            
            logging.info(f"LSTM模型加载成功: {model_path}")
            return model, model_info
            
        except Exception as e:
            logging.error(f"LSTM模型加载失败: {str(e)}")
            return None, None
    
    def prepare_prediction_data(self, df, features, seq_length=30):
        """为预测准备数据"""
        # 确保数据按时间排序
        if 'Date' in df.columns:
            df_sorted = df.sort_values('Date').reset_index(drop=True)
        else:
            df_sorted = df.copy()
        
        # 选择特征
        X_data = df_sorted[features].values
        dates = df_sorted['Date'].values if 'Date' in df_sorted.columns else range(len(df_sorted))
        
        # 检查和处理NaN值
        if np.isnan(X_data).any():
            X_data = pd.DataFrame(X_data).fillna(method='ffill').fillna(method='bfill').values
        
        # 标准化数据 (使用与训练时相同的方法)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_data)
        
        # 创建序列
        sequences = []
        sequence_dates = []
        
        for i in range(seq_length, len(X_scaled)):
            sequences.append(X_scaled[(i-seq_length):i])
            sequence_dates.append(dates[i])
        
        return np.array(sequences), sequence_dates, scaler
    
    def generate_volatility_predictions(self, model, sequences, scaler_y=None):
        """生成波动率预测"""
        try:
            predictions_scaled = model.predict(sequences, verbose=0)
            
            # 如果有目标变量的scaler，进行反标准化
            if scaler_y is not None:
                predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
            else:
                predictions = predictions_scaled.flatten()
            
            return predictions
            
        except Exception as e:
            logging.error(f"波动率预测失败: {str(e)}")
            return None
    
    def volatility_timing_strategy(self, df, volatility_predictions, prediction_dates, 
                                  vol_threshold_low=0.8, vol_threshold_high=1.2):
        """波动率择时策略
        
        策略逻辑:
        - 当预测波动率较低时，增加仓位 (低波动率时市场相对稳定)
        - 当预测波动率较高时，减少仓位 (高波动率时市场风险较大)
        """
        # 合并预测数据
        pred_df = pd.DataFrame({
            'Date': prediction_dates,
            'Predicted_Vol': volatility_predictions
        })
        
        # 与原数据合并
        strategy_df = df.merge(pred_df, on='Date', how='inner')
        
        if len(strategy_df) == 0:
            logging.error("预测数据与原数据无法匹配")
            return None
        
        # 计算波动率相对水平 (相对于历史均值)
        vol_mean = strategy_df['Predicted_Vol'].mean()
        vol_std = strategy_df['Predicted_Vol'].std()
        
        strategy_df['Vol_Zscore'] = (strategy_df['Predicted_Vol'] - vol_mean) / vol_std
        
        # 生成交易信号
        strategy_df['Position'] = 0.0
        
        # 低波动率时增加仓位
        low_vol_mask = strategy_df['Vol_Zscore'] < -vol_threshold_low
        strategy_df.loc[low_vol_mask, 'Position'] = 1.0
        
        # 中等波动率时保持中性仓位
        medium_vol_mask = (strategy_df['Vol_Zscore'] >= -vol_threshold_low) & (strategy_df['Vol_Zscore'] <= vol_threshold_high)
        strategy_df.loc[medium_vol_mask, 'Position'] = 0.5
        
        # 高波动率时减少仓位
        high_vol_mask = strategy_df['Vol_Zscore'] > vol_threshold_high
        strategy_df.loc[high_vol_mask, 'Position'] = 0.0
        
        return strategy_df
    
    def volatility_contrarian_strategy(self, df, volatility_predictions, prediction_dates):
        """波动率逆向策略
        
        策略逻辑:
        - 当预测波动率上升时，做空 (波动率回归)
        - 当预测波动率下降时，做多 (波动率回归)
        """
        # 合并预测数据
        pred_df = pd.DataFrame({
            'Date': prediction_dates,
            'Predicted_Vol': volatility_predictions
        })
        
        strategy_df = df.merge(pred_df, on='Date', how='inner')
        
        if len(strategy_df) == 0:
            return None
        
        # 计算波动率变化预测
        strategy_df['Vol_Change_Pred'] = strategy_df['Predicted_Vol'].diff()
        
        # 生成交易信号 (逆向)
        strategy_df['Position'] = 0.0
        
        # 预测波动率上升时做空
        vol_up_mask = strategy_df['Vol_Change_Pred'] > 0
        strategy_df.loc[vol_up_mask, 'Position'] = -0.5
        
        # 预测波动率下降时做多
        vol_down_mask = strategy_df['Vol_Change_Pred'] < 0
        strategy_df.loc[vol_down_mask, 'Position'] = 0.5
        
        return strategy_df
    
    def mean_reversion_strategy(self, df, volatility_predictions, prediction_dates, 
                               lookback_window=20):
        """均值回归策略
        
        基于波动率预测的均值回归策略
        """
        pred_df = pd.DataFrame({
            'Date': prediction_dates,
            'Predicted_Vol': volatility_predictions
        })
        
        strategy_df = df.merge(pred_df, on='Date', how='inner')
        
        if len(strategy_df) == 0:
            return None
        
        # 计算滚动均值和标准差
        strategy_df['Vol_MA'] = strategy_df['Predicted_Vol'].rolling(window=lookback_window).mean()
        strategy_df['Vol_Std'] = strategy_df['Predicted_Vol'].rolling(window=lookback_window).std()
        
        # 计算波动率偏离度
        strategy_df['Vol_Deviation'] = (strategy_df['Predicted_Vol'] - strategy_df['Vol_MA']) / strategy_df['Vol_Std']
        
        # 生成交易信号
        strategy_df['Position'] = np.tanh(-strategy_df['Vol_Deviation'] * 0.5)  # 使用tanh函数平滑仓位
        
        return strategy_df
    
    def calculate_strategy_returns(self, strategy_df, return_column='Returns'):
        """计算策略收益率"""
        if return_column not in strategy_df.columns:
            # 计算收益率 (假设有价格数据)
            price_cols = [col for col in strategy_df.columns if 'Close' in col or 'Price' in col]
            if price_cols:
                strategy_df[return_column] = strategy_df[price_cols[0]].pct_change()
            else:
                logging.error("无法找到价格或收益率数据")
                return None
        
        # 计算策略收益率
        strategy_df['Strategy_Returns'] = strategy_df['Position'].shift(1) * strategy_df[return_column]
        
        # 计算累计收益率
        strategy_df['Cumulative_Returns'] = (1 + strategy_df['Strategy_Returns']).cumprod() - 1
        strategy_df['Benchmark_Cumulative'] = (1 + strategy_df[return_column]).cumprod() - 1
        
        return strategy_df
    
    def run_backtest(self, df, model_path, strategy_type='timing', **strategy_params):
        """运行回测"""
        logging.info(f"开始运行{strategy_type}策略回测...")
        
        # 加载模型
        model, model_info = self.load_lstm_model(model_path)
        if model is None:
            return None
        
        # 获取特征列表
        features = model_info.get('features', [])
        seq_length = model_info.get('seq_length', 30)
        
        if not features:
            logging.error("无法获取模型特征信息")
            return None
        
        # 准备预测数据
        sequences, prediction_dates, scaler_X = self.prepare_prediction_data(df, features, seq_length)
        
        if len(sequences) == 0:
            logging.error("无法准备预测数据")
            return None
        
        # 生成波动率预测
        volatility_predictions = self.generate_volatility_predictions(model, sequences)
        
        if volatility_predictions is None:
            return None
        
        # 运行策略
        if strategy_type == 'timing':
            strategy_df = self.volatility_timing_strategy(
                df, volatility_predictions, prediction_dates, **strategy_params
            )
        elif strategy_type == 'contrarian':
            strategy_df = self.volatility_contrarian_strategy(
                df, volatility_predictions, prediction_dates
            )
        elif strategy_type == 'mean_reversion':
            strategy_df = self.mean_reversion_strategy(
                df, volatility_predictions, prediction_dates, **strategy_params
            )
        else:
            logging.error(f"未知策略类型: {strategy_type}")
            return None
        
        if strategy_df is None:
            return None
        
        # 计算策略收益率
        strategy_df = self.calculate_strategy_returns(strategy_df)
        
        if strategy_df is None:
            return None
        
        # 计算性能指标
        strategy_returns = strategy_df['Strategy_Returns'].dropna()
        benchmark_returns = strategy_df['Returns'].dropna()
        
        if len(strategy_returns) == 0:
            logging.error("策略收益率计算失败")
            return None
        
        # 创建结果字典
        result = {
            'strategy_type': strategy_type,
            'strategy_params': strategy_params,
            'strategy_data': strategy_df,
            'strategy_returns': strategy_returns,
            'benchmark_returns': benchmark_returns,
            'model_info': model_info
        }
        
        # 计算性能指标
        result['performance_metrics'] = self.backtest_utils.create_performance_metrics(
            strategy_returns, benchmark_returns
        )
        
        logging.info(f"{strategy_type}策略回测完成")
        logging.info(f"年化收益率: {result['performance_metrics']['年化收益率']:.4f}")
        logging.info(f"夏普比率: {result['performance_metrics']['夏普比率']:.4f}")
        
        return result
    
    def run_multiple_strategies(self, df, model_path_list, save_dir=None):
        """运行多个策略的回测"""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        all_results = {}
        
        for model_path in model_path_list:
            model_name = os.path.basename(model_path).replace('.h5', '')
            
            # 运行不同类型的策略
            strategies = [
                ('timing', {'vol_threshold_low': 0.8, 'vol_threshold_high': 1.2}),
                ('contrarian', {}),
                ('mean_reversion', {'lookback_window': 20})
            ]
            
            for strategy_type, params in strategies:
                strategy_key = f"{model_name}_{strategy_type}"
                
                result = self.run_backtest(df, model_path, strategy_type, **params)
                
                if result is not None:
                    all_results[strategy_key] = result
        
        self.strategy_results = all_results
        
        # 保存结果
        if save_dir:
            self.save_backtest_results(save_dir)
        
        return all_results
    
    def save_backtest_results(self, save_dir):
        """保存回测结果"""
        if not self.strategy_results:
            return
        
        # 创建性能指标汇总
        performance_summary = []
        for strategy_name, result in self.strategy_results.items():
            metrics = result['performance_metrics'].copy()
            metrics['策略名称'] = strategy_name
            metrics['策略类型'] = result['strategy_type']
            performance_summary.append(metrics)
        
        # 保存性能汇总
        summary_df = pd.DataFrame(performance_summary)
        summary_file = os.path.join(save_dir, 'strategy_performance_summary.csv')
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        
        # 保存各策略详细数据
        for strategy_name, result in self.strategy_results.items():
            strategy_data_file = os.path.join(save_dir, f'{strategy_name}_data.csv')
            result['strategy_data'].to_csv(strategy_data_file, index=False, encoding='utf-8-sig')
        
        # 绘制策略比较图
        strategy_returns_dict = {
            name: result['strategy_returns'] 
            for name, result in self.strategy_results.items()
        }
        
        if strategy_returns_dict:
            # 获取基准收益率
            benchmark_returns = list(self.strategy_results.values())[0]['benchmark_returns']
            
            comparison_plot = os.path.join(save_dir, 'strategy_comparison.png')
            self.backtest_utils.compare_strategies(
                strategy_returns_dict, benchmark_returns, comparison_plot
            )
        
        # 生成个别策略报告
        for strategy_name, result in self.strategy_results.items():
            strategy_dir = os.path.join(save_dir, strategy_name)
            self.backtest_utils.create_tear_sheet(
                result['strategy_returns'], 
                result['benchmark_returns'],
                strategy_name,
                strategy_dir
            )
        
        logging.info(f"回测结果保存至: {save_dir}")
    
    def analyze_strategy_attribution(self, strategy_result):
        """分析策略收益归因"""
        strategy_df = strategy_result['strategy_data']
        
        if 'Position' not in strategy_df.columns or 'Strategy_Returns' not in strategy_df.columns:
            return None
        
        # 分析不同仓位水平的收益贡献
        position_bins = pd.cut(strategy_df['Position'], bins=5, labels=['空仓', '轻仓', '中仓', '重仓', '满仓'])
        
        attribution = {}
        for position_level in position_bins.cat.categories:
            mask = position_bins == position_level
            if mask.sum() > 0:
                level_returns = strategy_df.loc[mask, 'Strategy_Returns']
                attribution[position_level] = {
                    '交易次数': mask.sum(),
                    '平均收益率': level_returns.mean(),
                    '胜率': (level_returns > 0).mean(),
                    '总贡献': level_returns.sum()
                }
        
        return attribution

def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 获取最新数据
    data_file = get_latest_merged_data()
    if not data_file:
        logging.error("未找到合并数据文件")
        return
    
    # 创建策略实例
    strategy = LSTMVolatilityStrategy()
    
    # 加载数据
    df = strategy.utils.load_merged_data(data_file)
    if df is None:
        logging.error("数据加载失败")
        return
    
    # 查找LSTM模型文件
    experiments_dir = Config.EXPERIMENTS_DIR
    if not os.path.exists(experiments_dir):
        logging.error("未找到实验目录")
        return
    
    # 搜索所有LSTM模型文件
    lstm_models = []
    for root, dirs, files in os.walk(experiments_dir):
        for file in files:
            if file.startswith('lstm_') and file.endswith('.h5'):
                lstm_models.append(os.path.join(root, file))
    
    if not lstm_models:
        logging.error("未找到LSTM模型文件")
        return
    
    logging.info(f"找到 {len(lstm_models)} 个LSTM模型")
    
    # 运行回测
    backtest_dir = os.path.join(experiments_dir, f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    results = strategy.run_multiple_strategies(df, lstm_models[:3], backtest_dir)  # 限制前3个模型
    
    print(f"\n回测完成，共测试 {len(results)} 个策略")
    print(f"结果保存在: {backtest_dir}")
    
    # 显示最佳策略
    if results:
        best_strategy = max(results.items(), key=lambda x: x[1]['performance_metrics']['夏普比率'])
        print(f"\n最佳策略: {best_strategy[0]}")
        print(f"夏普比率: {best_strategy[1]['performance_metrics']['夏普比率']:.4f}")
        print(f"年化收益率: {best_strategy[1]['performance_metrics']['年化收益率']:.4f}")

if __name__ == "__main__":
    main()
