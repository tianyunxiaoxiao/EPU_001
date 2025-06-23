"""
随机森林模型模块
使用随机森林进行回归预测
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
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

class RandomForestModels:
    """随机森林模型类"""
    
    def __init__(self, experiment_dir=None):
        self.utils = ModelUtils()
        self.experiment_dir = experiment_dir or setup_experiment_dir()
        self.results = []
        
    def get_param_grid(self, quick_search=False):
        """获取参数网格"""
        if quick_search:
            # 快速搜索参数
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }
        else:
            # 完整搜索参数
            param_grid = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.3, 0.5],
                'bootstrap': [True, False]
            }
        
        return param_grid
    
    def run_random_forest(self, X_train, X_test, y_train, y_test, feature_names, target_name, 
                         optimize_params=True, quick_search=False):
        """运行随机森林模型"""
        try:
            logging.info(f"开始训练随机森林模型 - {target_name}")
            
            # 基础随机森林模型
            base_rf = RandomForestRegressor(
                random_state=Config.MODEL_CONFIGS['random_state'],
                n_jobs=-1
            )
            
            if optimize_params:
                # 参数优化
                param_grid = self.get_param_grid(quick_search)
                
                if quick_search or len(X_train) < 1000:
                    # 网格搜索
                    rf_search = GridSearchCV(
                        base_rf, param_grid, 
                        cv=5, scoring='r2', n_jobs=-1,
                        verbose=1
                    )
                else:
                    # 随机搜索 (对于大数据集更高效)
                    rf_search = RandomizedSearchCV(
                        base_rf, param_grid,
                        n_iter=50, cv=5, scoring='r2', 
                        random_state=Config.MODEL_CONFIGS['random_state'],
                        n_jobs=-1, verbose=1
                    )
                
                rf_search.fit(X_train, y_train)
                best_rf = rf_search.best_estimator_
                best_params = rf_search.best_params_
                cv_score = rf_search.best_score_
                
                logging.info(f"参数优化完成 - {target_name}: CV Score = {cv_score:.4f}")
                
            else:
                # 使用默认参数
                best_rf = RandomForestRegressor(
                    n_estimators=100,
                    random_state=Config.MODEL_CONFIGS['random_state'],
                    n_jobs=-1
                )
                best_rf.fit(X_train, y_train)
                best_params = best_rf.get_params()
                cv_score = None
            
            # 预测
            y_train_pred = best_rf.predict(X_train)
            y_test_pred = best_rf.predict(X_test)
            
            # 评估
            train_metrics = self.utils.evaluate_model(y_train, y_train_pred, "RF_train")
            test_metrics = self.utils.evaluate_model(y_test, y_test_pred, "RF_test")
            
            # 特征重要性
            feature_importance = dict(zip(feature_names, best_rf.feature_importances_))
            
            # 交叉验证
            cv_results = self.utils.cross_validate_model(best_rf, X_train, y_train)
            
            model_summary = {
                'model_type': 'RandomForest',
                'target': target_name,
                'features': feature_names,
                'n_features': len(feature_names),
                'best_params': best_params,
                'cv_score': cv_score,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'feature_importance': feature_importance,
                'cv_results': cv_results,
                'n_trees': best_rf.n_estimators
            }
            
            # 保存模型
            model_file = os.path.join(self.experiment_dir, 'models', f'rf_{target_name}_{len(feature_names)}features.pkl')
            self.utils.save_model(best_rf, model_summary, model_file.replace('.pkl', '.json'))
            
            # 绘制预测图
            plot_file = os.path.join(self.experiment_dir, 'plots', f'rf_{target_name}_predictions.png')
            self.utils.plot_predictions(y_test, y_test_pred, f"Random Forest - {target_name}", plot_file)
            
            # 绘制特征重要性图
            importance_plot = os.path.join(self.experiment_dir, 'plots', f'rf_{target_name}_importance.png')
            importance_scores = list(feature_importance.values())
            self.utils.plot_feature_importance(
                importance_scores, feature_names, 
                f"随机森林特征重要性 - {target_name}", importance_plot
            )
            
            logging.info(f"随机森林模型完成 - {target_name}: R² = {test_metrics['r2_score']:.4f}")
            
            return model_summary
            
        except Exception as e:
            logging.error(f"随机森林模型失败 - {target_name}: {str(e)}")
            return None
    
    def run_all_rf_models(self, df, optimize_params=True, quick_search=False):
        """运行所有随机森林模型"""
        logging.info("开始运行随机森林模型...")
        
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
                
                # 随机森林不需要特征标准化，但可以选择性使用
                # X_train_scaled, X_test_scaled, scaler = self.utils.scale_features(X_train, X_test)
                
                # 运行随机森林模型
                rf_result = self.run_random_forest(
                    X_train, X_test, y_train, y_test,
                    pair_features, pair_target,
                    optimize_params, quick_search
                )
                
                if rf_result:
                    rf_result['feature_target_pair'] = pair_name
                    all_results.append(rf_result)
                    
            except Exception as e:
                logging.error(f"处理配对失败 {pair_name}: {str(e)}")
                continue
        
        self.results = all_results
        
        # 保存结果
        self.save_results()
        
        logging.info(f"随机森林模型完成，总计 {len(all_results)} 个模型")
        
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
                'model_name': f"RF_{result['target']}",
                'feature_target_pair': result.get('feature_target_pair', ''),
                'model_type': result['model_type'],
                'target': result['target'],
                'n_features': result['n_features'],
                'n_trees': result.get('n_trees', 0),
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
        summary_file = os.path.join(self.experiment_dir, 'results', 'rf_models_summary.csv')
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        
        # 保存详细结果
        detailed_file = os.path.join(self.experiment_dir, 'results', 'rf_models_detailed.json')
        import json
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
        
        # 绘制模型比较图
        comparison_plot = os.path.join(self.experiment_dir, 'plots', 'rf_models_comparison.png')
        self.utils.plot_model_comparison(summary_df, 'r2_score', comparison_plot)
        
        logging.info(f"随机森林模型结果保存完成")
        logging.info(f"汇总文件: {summary_file}")
        logging.info(f"详细结果: {detailed_file}")
        
        # 输出前5个最佳模型
        print("\n=== 随机森林模型性能排名 (前5名) ===")
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
        importance_analysis_file = os.path.join(self.experiment_dir, 'results', 'rf_feature_importance_analysis.csv')
        importance_df.to_csv(importance_analysis_file, index=False, encoding='utf-8-sig')
        
        # 绘制平均特征重要性图
        importance_plot_file = os.path.join(self.experiment_dir, 'plots', 'rf_avg_feature_importance.png')
        top_features = importance_df.head(20)  # 显示前20个特征
        self.utils.plot_feature_importance(
            top_features['avg_importance'].values,
            top_features['feature'].values,
            "随机森林平均特征重要性",
            importance_plot_file
        )
        
        logging.info(f"特征重要性分析完成: {importance_analysis_file}")
        
        return importance_df
    
    def plot_learning_curves(self, X, y, feature_names, target_name):
        """绘制学习曲线"""
        from sklearn.model_selection import learning_curve
        
        try:
            rf = RandomForestRegressor(
                n_estimators=100,
                random_state=Config.MODEL_CONFIGS['random_state'],
                n_jobs=-1
            )
            
            train_sizes, train_scores, val_scores = learning_curve(
                rf, X, y, cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='r2'
            )
            
            # 计算均值和标准差
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            # 绘图
            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, train_mean, 'o-', color='blue', label='训练分数')
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
            
            plt.plot(train_sizes, val_mean, 'o-', color='red', label='验证分数')
            plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
            
            plt.xlabel('训练样本数')
            plt.ylabel('R² 分数')
            plt.title(f'随机森林学习曲线 - {target_name}')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            
            # 保存图片
            curve_plot_file = os.path.join(self.experiment_dir, 'plots', f'rf_{target_name}_learning_curve.png')
            plt.savefig(curve_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"学习曲线保存至: {curve_plot_file}")
            
        except Exception as e:
            logging.error(f"绘制学习曲线失败 - {target_name}: {str(e)}")

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
    rf_models = RandomForestModels()
    
    # 加载数据
    df = rf_models.utils.load_merged_data(data_file)
    if df is None:
        logging.error("数据加载失败")
        return
    
    # 运行所有随机森林模型 (使用快速搜索以节省时间)
    results = rf_models.run_all_rf_models(df, optimize_params=True, quick_search=True)
    
    # 分析特征重要性
    importance_df = rf_models.analyze_feature_importance()
    
    print(f"\n随机森林模型实验完成，共训练 {len(results)} 个模型")
    print(f"结果保存在: {rf_models.experiment_dir}")
    
    if importance_df is not None and len(importance_df) > 0:
        print("\n=== 最重要的前10个特征 ===")
        top10_features = importance_df.head(10)
        for idx, row in top10_features.iterrows():
            print(f"{idx+1}. {row['feature']}: {row['avg_importance']:.4f}")

if __name__ == "__main__":
    main()
