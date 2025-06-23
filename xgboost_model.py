"""
XGBoost模型模块
使用XGBoost进行回归预测
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import os
import sys
import logging
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config
from src.modeling.model_utils import ModelUtils, get_latest_merged_data, setup_experiment_dir

class XGBoostModels:
    """XGBoost模型类"""
    
    def __init__(self, experiment_dir=None):
        self.utils = ModelUtils()
        self.experiment_dir = experiment_dir or setup_experiment_dir()
        self.results = []
        
    def get_param_grid(self, quick_search=False):
        """获取参数网格"""
        if quick_search:
            # 快速搜索参数
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        else:
            # 完整搜索参数
            param_grid = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 4, 5, 6, 7, 8],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [0, 0.1, 0.5, 1.0]
            }
        
        return param_grid
    
    def run_xgboost(self, X_train, X_test, y_train, y_test, feature_names, target_name, 
                    optimize_params=True, quick_search=False):
        """运行XGBoost模型"""
        try:
            logging.info(f"开始训练XGBoost模型 - {target_name}")
            
            # 基础XGBoost模型
            base_xgb = xgb.XGBRegressor(
                random_state=Config.MODEL_CONFIGS['random_state'],
                n_jobs=-1,
                verbosity=0
            )
            
            if optimize_params:
                # 参数优化
                param_grid = self.get_param_grid(quick_search)
                
                if quick_search or len(X_train) < 1000:
                    # 网格搜索
                    xgb_search = GridSearchCV(
                        base_xgb, param_grid, 
                        cv=5, scoring='r2', n_jobs=-1,
                        verbose=1
                    )
                else:
                    # 随机搜索 (对于大数据集更高效)
                    xgb_search = RandomizedSearchCV(
                        base_xgb, param_grid,
                        n_iter=50, cv=5, scoring='r2', 
                        random_state=Config.MODEL_CONFIGS['random_state'],
                        n_jobs=-1, verbose=1
                    )
                
                xgb_search.fit(X_train, y_train)
                best_xgb = xgb_search.best_estimator_
                best_params = xgb_search.best_params_
                cv_score = xgb_search.best_score_
                
                logging.info(f"参数优化完成 - {target_name}: CV Score = {cv_score:.4f}")
                
            else:
                # 使用默认参数
                best_xgb = xgb.XGBRegressor(
                    n_estimators=100,
                    random_state=Config.MODEL_CONFIGS['random_state'],
                    n_jobs=-1,
                    verbosity=0
                )
                best_xgb.fit(X_train, y_train)
                best_params = best_xgb.get_params()
                cv_score = None
            
            # 预测
            y_train_pred = best_xgb.predict(X_train)
            y_test_pred = best_xgb.predict(X_test)
            
            # 评估
            train_metrics = self.utils.evaluate_model(y_train, y_train_pred, "XGB_train")
            test_metrics = self.utils.evaluate_model(y_test, y_test_pred, "XGB_test")
            
            # 特征重要性
            feature_importance = dict(zip(feature_names, best_xgb.feature_importances_))
            
            # 交叉验证
            cv_results = self.utils.cross_validate_model(best_xgb, X_train, y_train)
            
            model_summary = {
                'model_type': 'XGBoost',
                'target': target_name,
                'features': feature_names,
                'n_features': len(feature_names),
                'best_params': best_params,
                'cv_score': cv_score,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'feature_importance': feature_importance,
                'cv_results': cv_results,
                'n_estimators': best_xgb.n_estimators
            }
            
            # 保存模型
            model_file = os.path.join(self.experiment_dir, 'models', f'xgb_{target_name}_{len(feature_names)}features.pkl')
            self.utils.save_model(best_xgb, model_summary, model_file.replace('.pkl', '.json'))
            
            # 绘制预测图
            plot_file = os.path.join(self.experiment_dir, 'plots', f'xgb_{target_name}_predictions.png')
            self.utils.plot_predictions(y_test, y_test_pred, f"XGBoost - {target_name}", plot_file)
            
            # 绘制特征重要性图
            importance_plot = os.path.join(self.experiment_dir, 'plots', f'xgb_{target_name}_importance.png')
            importance_scores = list(feature_importance.values())
            self.utils.plot_feature_importance(
                importance_scores, feature_names, 
                f"XGBoost特征重要性 - {target_name}", importance_plot
            )
            
            # 绘制训练过程中的损失曲线
            if hasattr(best_xgb, 'evals_result_') and best_xgb.evals_result_:
                self.plot_training_curves(best_xgb, target_name)
            
            logging.info(f"XGBoost模型完成 - {target_name}: R² = {test_metrics['r2_score']:.4f}")
            
            return model_summary
            
        except Exception as e:
            logging.error(f"XGBoost模型失败 - {target_name}: {str(e)}")
            return None
    
    def plot_training_curves(self, model, target_name):
        """绘制训练曲线"""
        try:
            if not hasattr(model, 'evals_result_') or not model.evals_result_:
                return
            
            results = model.evals_result_
            epochs = len(results['validation_0']['rmse'])
            x_axis = range(0, epochs)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
            if 'validation_1' in results:
                ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
            ax.legend()
            ax.set_ylabel('RMSE')
            ax.set_xlabel('Epoch')
            ax.set_title(f'XGBoost训练曲线 - {target_name}')
            ax.grid(True, alpha=0.3)
            
            # 保存图片
            curve_plot_file = os.path.join(self.experiment_dir, 'plots', f'xgb_{target_name}_training_curve.png')
            plt.savefig(curve_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"训练曲线保存至: {curve_plot_file}")
            
        except Exception as e:
            logging.error(f"绘制训练曲线失败 - {target_name}: {str(e)}")
    
    def run_all_xgboost_models(self, df, optimize_params=True, quick_search=False):
        """运行所有XGBoost模型"""
        logging.info("开始运行XGBoost模型...")
        
        # 准备特征和目标
        features, targets = self.utils.prepare_features_targets(df)
        
        if not features or not targets:
            logging.error("没有找到有效的特征或目标变量")
            return []
        
        # 创建特征-目标配对
        pairs = self.utils.create_feature_target_pairs(df, features, targets)
        
        all_results = []
        
        for pair in pairs:
            pair_features = pair['features']
            pair_target = pair['target']
            pair_name = pair['name']
            
            logging.info(f"处理配对: {pair_name}")
            
            try:
                # 分割数据
                X_train, X_test, y_train, y_test = self.utils.split_data(
                    df, pair_features, pair_target
                )
                
                if len(X_train) == 0 or len(X_test) == 0:
                    logging.warning(f"数据分割后无有效样本: {pair_name}")
                    continue
                
                # XGBoost通常不需要特征标准化，但可以选择性使用
                # X_train_scaled, X_test_scaled, scaler = self.utils.scale_features(X_train, X_test)
                
                # 运行XGBoost模型
                xgb_result = self.run_xgboost(
                    X_train, X_test, y_train, y_test,
                    pair_features, pair_target,
                    optimize_params, quick_search
                )
                
                if xgb_result:
                    xgb_result['feature_target_pair'] = pair_name
                    all_results.append(xgb_result)
                    
            except Exception as e:
                logging.error(f"处理配对失败 {pair_name}: {str(e)}")
                continue
        
        self.results = all_results
        
        # 保存结果
        self.save_results()
        
        logging.info(f"XGBoost模型完成，总计 {len(all_results)} 个模型")
        
        return all_results
    
    def save_results(self):
        """保存模型结果"""
        if not self.results:
            return
        
        # 提取主要指标用于汇总
        summary_data = []
        for result in self.results:
            test_metrics = result.get('test_metrics', {})
            cv_results = result.get('cv_results', {})
            summary_data.append({
                'model_name': f"XGB_{result['target']}",
                'feature_target_pair': result.get('feature_target_pair', ''),
                'model_type': result['model_type'],
                'target': result['target'],
                'n_features': result['n_features'],
                'n_estimators': result.get('n_estimators', 0),
                'r2_score': test_metrics.get('r2_score', 0),
                'rmse': test_metrics.get('rmse', 0),
                'mae': test_metrics.get('mae', 0),
                'mape': test_metrics.get('mape', 0),
                'cv_mean': cv_results.get('cv_mean', 0),
                'cv_std': cv_results.get('cv_std', 0),
                'n_samples': test_metrics.get('n_samples', 0)
            })
        
        # 创建汇总表
        summary_df = self.utils.create_model_summary(summary_data)
        
        # 保存汇总表
        summary_file = os.path.join(self.experiment_dir, 'results', 'xgb_models_summary.csv')
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        
        # 保存详细结果
        detailed_file = os.path.join(self.experiment_dir, 'results', 'xgb_models_detailed.json')
        import json
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
        
        # 绘制模型比较图
        comparison_plot = os.path.join(self.experiment_dir, 'plots', 'xgb_models_comparison.png')
        self.utils.plot_model_comparison(summary_df, 'r2_score', comparison_plot)
        
        logging.info(f"XGBoost模型结果保存完成")
        logging.info(f"汇总文件: {summary_file}")
        logging.info(f"详细结果: {detailed_file}")
        
        # 输出前5个最佳模型
        print("\n=== XGBoost模型性能排名 (前5名) ===")
        top5 = summary_df.head(5)
        for idx, row in top5.iterrows():
            print(f"{row['rank']}. {row['model_name']}: R² = {row['r2_score']:.4f}, RMSE = {row['rmse']:.6f}")
    
    def analyze_feature_importance(self):
        """分析特征重要性"""
        if not self.results:
            logging.warning("没有模型结果可分析")
            return
        
        # 汇总所有模型的特征重要性
        feature_importance_summary = {}
        
        for result in self.results:
            feature_importance = result.get('feature_importance', {})
            target = result['target']
            
            for feature, importance in feature_importance.items():
                if feature not in feature_importance_summary:
                    feature_importance_summary[feature] = {
                        'total_importance': 0,
                        'count': 0,
                        'targets': []
                    }
                
                feature_importance_summary[feature]['total_importance'] += importance
                feature_importance_summary[feature]['count'] += 1
                feature_importance_summary[feature]['targets'].append(target)
        
        # 计算平均重要性
        feature_avg_importance = []
        for feature, data in feature_importance_summary.items():
            avg_importance = data['total_importance'] / data['count']
            feature_avg_importance.append({
                'feature': feature,
                'avg_importance': avg_importance,
                'appearance_count': data['count'],
                'targets': list(set(data['targets']))
            })
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame(feature_avg_importance)
        importance_df = importance_df.sort_values('avg_importance', ascending=False).reset_index(drop=True)
        
        # 保存特征重要性分析
        importance_analysis_file = os.path.join(self.experiment_dir, 'results', 'xgb_feature_importance_analysis.csv')
        importance_df.to_csv(importance_analysis_file, index=False, encoding='utf-8-sig')
        
        # 绘制平均特征重要性图
        importance_plot_file = os.path.join(self.experiment_dir, 'plots', 'xgb_avg_feature_importance.png')
        top_features = importance_df.head(20)  # 显示前20个特征
        self.utils.plot_feature_importance(
            top_features['avg_importance'].values,
            top_features['feature'].values,
            "XGBoost平均特征重要性",
            importance_plot_file
        )
        
        logging.info(f"特征重要性分析完成: {importance_analysis_file}")
        
        return importance_df
    
    def plot_shap_analysis(self, model, X_sample, feature_names, target_name, n_samples=100):
        """SHAP分析"""
        try:
            import shap
            
            # 创建SHAP解释器
            explainer = shap.TreeExplainer(model)
            
            # 选择样本进行分析
            if len(X_sample) > n_samples:
                X_shap = X_sample.sample(n_samples, random_state=Config.MODEL_CONFIGS['random_state'])
            else:
                X_shap = X_sample
            
            # 计算SHAP值
            shap_values = explainer.shap_values(X_shap)
            
            # 绘制SHAP总结图
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False)
            shap_plot_file = os.path.join(self.experiment_dir, 'plots', f'xgb_{target_name}_shap_summary.png')
            plt.savefig(shap_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 绘制SHAP特征重要性图
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_shap, feature_names=feature_names, plot_type="bar", show=False)
            shap_importance_file = os.path.join(self.experiment_dir, 'plots', f'xgb_{target_name}_shap_importance.png')
            plt.savefig(shap_importance_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"SHAP分析完成 - {target_name}")
            logging.info(f"SHAP总结图: {shap_plot_file}")
            logging.info(f"SHAP重要性图: {shap_importance_file}")
            
        except ImportError:
            logging.warning("SHAP包未安装，跳过SHAP分析")
        except Exception as e:
            logging.error(f"SHAP分析失败 - {target_name}: {str(e)}")

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
    xgb_models = XGBoostModels()
    
    # 加载数据
    df = xgb_models.utils.load_merged_data(data_file)
    if df is None:
        logging.error("数据加载失败")
        return
    
    # 运行所有XGBoost模型 (使用快速搜索以节省时间)
    results = xgb_models.run_all_xgboost_models(df, optimize_params=True, quick_search=True)
    
    # 分析特征重要性
    importance_df = xgb_models.analyze_feature_importance()
    
    print(f"\nXGBoost模型实验完成，共训练 {len(results)} 个模型")
    print(f"结果保存在: {xgb_models.experiment_dir}")
    
    if importance_df is not None and len(importance_df) > 0:
        print("\n=== 最重要的前10个特征 ===")
        top10_features = importance_df.head(10)
        for idx, row in top10_features.iterrows():
            print(f"{idx+1}. {row['feature']}: {row['avg_importance']:.4f}")

if __name__ == "__main__":
    main()
