"""
LSTM深度学习模型模块
使用LSTM进行时间序列预测
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import os
import sys
import logging
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config
from src.modeling.model_utils import ModelUtils, get_latest_merged_data, setup_experiment_dir

class LSTMModels:
    """LSTM模型类"""
    
    def __init__(self, experiment_dir=None):
        self.utils = ModelUtils()
        self.experiment_dir = experiment_dir or setup_experiment_dir()
        self.results = []
        
        # 设置TensorFlow日志级别
        tf.get_logger().setLevel('ERROR')
        
        # 设置随机种子
        tf.random.set_seed(Config.MODEL_CONFIGS['random_state'])
        np.random.seed(Config.MODEL_CONFIGS['random_state'])
        
    def create_sequences(self, data, seq_length=30):
        """创建时间序列序列"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    
    def prepare_lstm_data(self, df, features, target, seq_length=30):
        """为LSTM准备数据"""
        # 确保数据按时间排序
        if 'Date' in df.columns:
            df_sorted = df.sort_values('Date').reset_index(drop=True)
        else:
            df_sorted = df.copy()
        
        # 选择特征和目标
        X_data = df_sorted[features].values
        y_data = df_sorted[target].values
        
        # 检查NaN值
        if np.isnan(X_data).any() or np.isnan(y_data).any():
            # 填充NaN值
            X_data = pd.DataFrame(X_data).fillna(method='ffill').fillna(method='bfill').values
            y_data = pd.Series(y_data).fillna(method='ffill').fillna(method='bfill').values
        
        # 标准化数据
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_scaled = scaler_X.fit_transform(X_data)
        y_scaled = scaler_y.fit_transform(y_data.reshape(-1, 1)).flatten()
        
        # 创建序列
        X_seq, y_seq = self.create_sequences(
            np.column_stack([X_scaled, y_scaled]), seq_length
        )
        
        if len(X_seq) == 0:
            return None, None, None, None, None, None
        
        # 分离特征和目标
        X_sequences = X_seq[:, :, :-1]  # 所有特征
        y_sequences = X_seq[:, -1, -1]  # 目标变量的最后一个值
        
        # 分割训练集和测试集
        train_size = int(len(X_sequences) * 0.8)
        
        X_train = X_sequences[:train_size]
        X_test = X_sequences[train_size:]
        y_train = y_sequences[:train_size]
        y_test = y_sequences[train_size:]
        
        return X_train, X_test, y_train, y_test, scaler_X, scaler_y
    
    def build_lstm_model(self, input_shape, lstm_units=[50, 50], dropout_rate=0.2):
        """构建LSTM模型"""
        model = Sequential()
        
        # 第一层LSTM
        model.add(LSTM(
            lstm_units[0], 
            return_sequences=len(lstm_units) > 1,
            input_shape=input_shape
        ))
        model.add(Dropout(dropout_rate))
        
        # 添加更多LSTM层
        for i in range(1, len(lstm_units)):
            return_seq = i < len(lstm_units) - 1
            model.add(LSTM(lstm_units[i], return_sequences=return_seq))
            model.add(Dropout(dropout_rate))
        
        # 输出层
        model.add(Dense(1))
        
        # 编译模型
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def run_lstm(self, X_train, X_test, y_train, y_test, scaler_y, 
                 feature_names, target_name, seq_length=30):
        """运行LSTM模型"""
        try:
            logging.info(f"开始训练LSTM模型 - {target_name}")
            
            # 构建模型
            input_shape = (X_train.shape[1], X_train.shape[2])
            model = self.build_lstm_model(input_shape)
            
            # 设置回调函数
            callbacks = [
                EarlyStopping(
                    patience=20, 
                    restore_best_weights=True,
                    monitor='val_loss'
                ),
                ReduceLROnPlateau(
                    patience=10,
                    factor=0.5,
                    min_lr=1e-7,
                    monitor='val_loss'
                )
            ]
            
            # 训练模型
            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=0
            )
            
            # 预测
            y_train_pred_scaled = model.predict(X_train, verbose=0)
            y_test_pred_scaled = model.predict(X_test, verbose=0)
            
            # 反标准化
            y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
            y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
            y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
            y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
            
            # 评估
            train_metrics = self.utils.evaluate_model(y_train_actual, y_train_pred, "LSTM_train")
            test_metrics = self.utils.evaluate_model(y_test_actual, y_test_pred, "LSTM_test")
            
            model_summary = {
                'model_type': 'LSTM',
                'target': target_name,
                'features': feature_names,
                'n_features': len(feature_names),
                'seq_length': seq_length,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'predictions': y_test_pred,  # 添加预测结果
                'history': {
                    'loss': history.history['loss'],
                    'val_loss': history.history['val_loss'],
                    'mae': history.history['mae'],
                    'val_mae': history.history['val_mae']
                },
                'epochs_trained': len(history.history['loss'])
            }
            
            # 保存模型
            model_file = os.path.join(self.experiment_dir, 'models', f'lstm_{target_name}_{len(feature_names)}features.h5')
            model.save(model_file)
            
            # 保存模型信息
            info_file = model_file.replace('.h5', '.json')
            import json
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(model_summary, f, ensure_ascii=False, indent=2, default=str)
            
            # 绘制预测图
            plot_file = os.path.join(self.experiment_dir, 'plots', f'lstm_{target_name}_predictions.png')
            self.utils.plot_predictions(y_test_actual, y_test_pred, f"LSTM - {target_name}", plot_file)
            
            # 绘制训练历史
            self.plot_training_history(history, target_name)
            
            logging.info(f"LSTM模型完成 - {target_name}: R² = {test_metrics['r2_score']:.4f}")
            
            return model_summary
            
        except Exception as e:
            logging.error(f"LSTM模型失败 - {target_name}: {str(e)}")
            return None
    
    def plot_training_history(self, history, target_name):
        """绘制训练历史"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 损失曲线
            ax1.plot(history.history['loss'], label='训练损失')
            ax1.plot(history.history['val_loss'], label='验证损失')
            ax1.set_title(f'LSTM训练损失 - {target_name}')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # MAE曲线
            ax2.plot(history.history['mae'], label='训练MAE')
            ax2.plot(history.history['val_mae'], label='验证MAE')
            ax2.set_title(f'LSTM训练MAE - {target_name}')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            history_plot_file = os.path.join(self.experiment_dir, 'plots', f'lstm_{target_name}_history.png')
            plt.savefig(history_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"训练历史保存至: {history_plot_file}")
            
        except Exception as e:
            logging.error(f"绘制训练历史失败 - {target_name}: {str(e)}")
    
    def run_all_lstm_models(self, df, seq_length=30):
        """运行所有LSTM模型"""
        logging.info("开始运行LSTM模型...")
        
        # 准备特征和目标
        features, targets = self.utils.prepare_features_targets(df)
        
        if not features or not targets:
            logging.error("没有找到有效的特征或目标变量")
            return []
        
        # 创建特征-目标配对 (LSTM通常使用所有特征)
        pairs = []
        for target in targets:
            pairs.append({
                'features': features,
                'target': target,
                'name': f"all_features_to_{target}",
                'type': 'lstm_full'
            })
        
        all_results = []
        
        for pair in pairs:
            pair_features = pair['features']
            pair_target = pair['target']
            pair_name = pair['name']
            
            logging.info(f"处理配对: {pair_name}")
            
            try:
                # 为LSTM准备数据
                X_train, X_test, y_train, y_test, scaler_X, scaler_y = self.prepare_lstm_data(
                    df, pair_features, pair_target, seq_length
                )
                
                if X_train is None or len(X_train) == 0:
                    logging.warning(f"LSTM数据准备失败: {pair_name}")
                    continue
                
                # 运行LSTM模型
                lstm_result = self.run_lstm(
                    X_train, X_test, y_train, y_test, scaler_y,
                    pair_features, pair_target, seq_length
                )
                
                if lstm_result:
                    lstm_result['feature_target_pair'] = pair_name
                    all_results.append(lstm_result)
                    
            except Exception as e:
                logging.error(f"处理配对失败 {pair_name}: {str(e)}")
                continue
        
        self.results = all_results
        
        # 保存结果
        self.save_results()
        
        logging.info(f"LSTM模型完成，总计 {len(all_results)} 个模型")
        
        return all_results
    
    def save_results(self):
        """保存模型结果"""
        if not self.results:
            return
        
        # 提取主要指标用于汇总
        summary_data = []
        for result in self.results:
            test_metrics = result.get('test_metrics', {})
            summary_data.append({
                'model_name': f"LSTM_{result['target']}",
                'feature_target_pair': result.get('feature_target_pair', ''),
                'model_type': result['model_type'],
                'target': result['target'],
                'n_features': result['n_features'],
                'seq_length': result.get('seq_length', 0),
                'epochs_trained': result.get('epochs_trained', 0),
                'r2_score': test_metrics.get('r2_score', 0),
                'rmse': test_metrics.get('rmse', 0),
                'mae': test_metrics.get('mae', 0),
                'mape': test_metrics.get('mape', 0),
                'n_samples': test_metrics.get('n_samples', 0)
            })
        
        # 创建汇总表
        summary_df = self.utils.create_model_summary(summary_data)
        
        # 保存汇总表
        summary_file = os.path.join(self.experiment_dir, 'results', 'lstm_models_summary.csv')
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        
        # 保存详细结果
        detailed_file = os.path.join(self.experiment_dir, 'results', 'lstm_models_detailed.json')
        import json
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
        
        # 绘制模型比较图
        comparison_plot = os.path.join(self.experiment_dir, 'plots', 'lstm_models_comparison.png')
        self.utils.plot_model_comparison(summary_df, 'r2_score', comparison_plot)
        
        logging.info(f"LSTM模型结果保存完成")
        logging.info(f"汇总文件: {summary_file}")
        logging.info(f"详细结果: {detailed_file}")
        
        # 输出前5个最佳模型
        print("\n=== LSTM模型性能排名 (前5名) ===")
        top5 = summary_df.head(5)
        for idx, row in top5.iterrows():
            print(f"{row['rank']}. {row['model_name']}: R² = {row['r2_score']:.4f}, RMSE = {row['rmse']:.6f}")
    
    def predict_future(self, model_file, scaler_X, scaler_y, last_sequence, n_steps=30):
        """使用训练好的模型预测未来"""
        try:
            # 加载模型
            model = tf.keras.models.load_model(model_file)
            
            # 预测未来步数
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(n_steps):
                # 预测下一步
                pred_scaled = model.predict(current_sequence.reshape(1, *current_sequence.shape), verbose=0)
                pred_actual = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
                predictions.append(pred_actual)
                
                # 更新序列 (简单方法：移除第一个时间步，添加预测值)
                # 这里需要根据具体情况调整
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1, -1] = pred_scaled[0, 0]  # 假设目标是最后一列
            
            return predictions
            
        except Exception as e:
            logging.error(f"未来预测失败: {str(e)}")
            return None

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
    
    # 创建模型实例
    lstm_models = LSTMModels()
    
    # 加载数据
    df = lstm_models.utils.load_merged_data(data_file)
    if df is None:
        logging.error("数据加载失败")
        return
    
    # 检查数据量是否足够进行LSTM训练
    if len(df) < 100:
        logging.error("数据量不足，无法进行LSTM训练")
        return
    
    # 运行所有LSTM模型
    seq_length = min(30, len(df) // 4)  # 自适应序列长度
    results = lstm_models.run_all_lstm_models(df, seq_length=seq_length)
    
    print(f"\nLSTM模型实验完成，共训练 {len(results)} 个模型")
    print(f"结果保存在: {lstm_models.experiment_dir}")

if __name__ == "__main__":
    main()
