"""
线性模型模块
包含OLS回归和LASSO正则化回归模型
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
import os
import sys
import logging
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config
from src.modeling.model_utils import ModelUtils, get_latest_merged_data, setup_experiment_dir

class LinearModels:
    """线性模型类"""
    
    def __init__(self, experiment_dir=None):
        self.utils = ModelUtils()
        self.experiment_dir = experiment_dir or setup_experiment_dir()
        self.results = []
        
    def run_ols_regression(self, X_train, X_test, y_train, y_test, feature_names, target_name):
        """运行OLS回归"""
        try:
            # 添加常数项
            X_train_sm = sm.add_constant(X_train)
            X_test_sm = sm.add_constant(X_test)
            
            # 拟合OLS模型
            ols_model = sm.OLS(y_train, X_train_sm).fit()
            
            # 预测
            y_train_pred = ols_model.predict(X_train_sm)
            y_test_pred = ols_model.predict(X_test_sm)
            
            # 评估
            train_metrics = self.utils.evaluate_model(y_train, y_train_pred, "OLS_train")
            test_metrics = self.utils.evaluate_model(y_test, y_test_pred, "OLS_test")
            
            # 获取统计信息
            model_summary = {
                'model_type': 'OLS',
                'target': target_name,
                'features': feature_names,
                'n_features': len(feature_names),
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'model_summary': str(ols_model.summary()),
                'coefficients': dict(zip(['const'] + feature_names, ols_model.params)),
                'pvalues': dict(zip(['const'] + feature_names, ols_model.pvalues)),
                'aic': ols_model.aic,
                'bic': ols_model.bic,
                'rsquared': ols_model.rsquared,
                'rsquared_adj': ols_model.rsquared_adj,
                'fvalue': ols_model.fvalue,
                'f_pvalue': ols_model.f_pvalue
            }
            
            # 保存模型
            model_file = os.path.join(self.experiment_dir, 'models', f'ols_{target_name}_{len(feature_names)}features.json')
            self.utils.save_model(ols_model, model_summary, model_file)
            
            # 绘制预测图
            plot_file = os.path.join(self.experiment_dir, 'plots', f'ols_{target_name}_predictions.png')
            self.utils.plot_predictions(y_test, y_test_pred, f"OLS - {target_name}", plot_file)
            
            logging.info(f"OLS模型完成 - {target_name}: R² = {test_metrics['r2_score']:.4f}")
            
            return model_summary
            
        except Exception as e:
            logging.error(f"OLS回归失败 - {target_name}: {str(e)}")
            return None
    
    def run_lasso_regression(self, X_train, X_test, y_train, y_test, feature_names, target_name):
        """运行LASSO回归"""
        try:
            # 使用交叉验证选择最优alpha
            lasso_cv = LassoCV(cv=5, random_state=Config.MODEL_CONFIGS['random_state'], max_iter=2000)
            lasso_cv.fit(X_train, y_train)
            
            best_alpha = lasso_cv.alpha_
            
            # 使用最优alpha训练最终模型
            lasso_model = Lasso(alpha=best_alpha, random_state=Config.MODEL_CONFIGS['random_state'], max_iter=2000)
            lasso_model.fit(X_train, y_train)
            
            # 预测
            y_train_pred = lasso_model.predict(X_train)
            y_test_pred = lasso_model.predict(X_test)
            
            # 评估
            train_metrics = self.utils.evaluate_model(y_train, y_train_pred, "LASSO_train")
            test_metrics = self.utils.evaluate_model(y_test, y_test_pred, "LASSO_test")
            
            # 获取特征选择信息
            selected_features = []
            feature_coeffs = {}
            for i, (feature, coef) in enumerate(zip(feature_names, lasso_model.coef_)):
                feature_coeffs[feature] = coef
                if abs(coef) > 1e-6:  # 非零系数
                    selected_features.append(feature)
            
            model_summary = {
                'model_type': 'LASSO',
                'target': target_name,
                'features': feature_names,
                'n_features': len(feature_names),
                'selected_features': selected_features,
                'n_selected_features': len(selected_features),
                'alpha': best_alpha,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'coefficients': feature_coeffs,
                'intercept': lasso_model.intercept_
            }
            
            # 保存模型
            model_file = os.path.join(self.experiment_dir, 'models', f'lasso_{target_name}_{len(feature_names)}features.pkl')
            self.utils.save_model(lasso_model, model_summary, model_file.replace('.pkl', '.json'))
            
            # 绘制预测图
            plot_file = os.path.join(self.experiment_dir, 'plots', f'lasso_{target_name}_predictions.png')
            self.utils.plot_predictions(y_test, y_test_pred, f"LASSO - {target_name}", plot_file)
            
            # 绘制特征系数图
            if len(selected_features) > 0:
                coef_plot_file = os.path.join(self.experiment_dir, 'plots', f'lasso_{target_name}_coefficients.png')
                selected_coefs = [feature_coeffs[f] for f in selected_features]
                self.utils.plot_feature_importance(
                    selected_coefs, selected_features, 
                    f"LASSO系数 - {target_name}", coef_plot_file
                )
            
            logging.info(f"LASSO模型完成 - {target_name}: R² = {test_metrics['r2_score']:.4f}, 选择特征数 = {len(selected_features)}")
            
            return model_summary
            
        except Exception as e:
            logging.error(f"LASSO回归失败 - {target_name}: {str(e)}")
            return None
    
    def run_ridge_regression(self, X_train, X_test, y_train, y_test, feature_names, target_name):
        """运行Ridge回归"""
        try:
            # 使用交叉验证选择最优alpha
            ridge_cv = RidgeCV(cv=5, alphas=np.logspace(-6, 6, 13))
            ridge_cv.fit(X_train, y_train)
            
            best_alpha = ridge_cv.alpha_
            
            # 使用最优alpha训练最终模型
            ridge_model = Ridge(alpha=best_alpha, random_state=Config.MODEL_CONFIGS['random_state'])
            ridge_model.fit(X_train, y_train)
            
            # 预测
            y_train_pred = ridge_model.predict(X_train)
            y_test_pred = ridge_model.predict(X_test)
            
            # 评估
            train_metrics = self.utils.evaluate_model(y_train, y_train_pred, "Ridge_train")
            test_metrics = self.utils.evaluate_model(y_test, y_test_pred, "Ridge_test")
            
            # 获取系数信息
            feature_coeffs = dict(zip(feature_names, ridge_model.coef_))
            
            model_summary = {
                'model_type': 'Ridge',
                'target': target_name,
                'features': feature_names,
                'n_features': len(feature_names),
                'alpha': best_alpha,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'coefficients': feature_coeffs,
                'intercept': ridge_model.intercept_
            }
            
            # 保存模型
            model_file = os.path.join(self.experiment_dir, 'models', f'ridge_{target_name}_{len(feature_names)}features.pkl')
            self.utils.save_model(ridge_model, model_summary, model_file.replace('.pkl', '.json'))
            
            # 绘制预测图
            plot_file = os.path.join(self.experiment_dir, 'plots', f'ridge_{target_name}_predictions.png')
            self.utils.plot_predictions(y_test, y_test_pred, f"Ridge - {target_name}", plot_file)
            
            logging.info(f"Ridge模型完成 - {target_name}: R² = {test_metrics['r2_score']:.4f}")
            
            return model_summary
            
        except Exception as e:
            logging.error(f"Ridge回归失败 - {target_name}: {str(e)}")
            return None
    
    def run_all_linear_models(self, df):
        """运行所有线性模型"""
        logging.info("开始运行线性模型...")
        
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
                
                # 标准化特征
                X_train_scaled, X_test_scaled, scaler = self.utils.scale_features(X_train, X_test)
                
                # 运行OLS回归 (仅对特征数较少的情况)
                if len(pair_features) <= 20:
                    ols_result = self.run_ols_regression(
                        X_train_scaled, X_test_scaled, y_train, y_test,
                        pair_features, pair_target
                    )
                    if ols_result:
                        ols_result['feature_target_pair'] = pair_name
                        all_results.append(ols_result)
                
                # 运行LASSO回归
                lasso_result = self.run_lasso_regression(
                    X_train_scaled, X_test_scaled, y_train, y_test,
                    pair_features, pair_target
                )
                if lasso_result:
                    lasso_result['feature_target_pair'] = pair_name
                    all_results.append(lasso_result)
                
                # 运行Ridge回归
                ridge_result = self.run_ridge_regression(
                    X_train_scaled, X_test_scaled, y_train, y_test,
                    pair_features, pair_target
                )
                if ridge_result:
                    ridge_result['feature_target_pair'] = pair_name
                    all_results.append(ridge_result)
                    
            except Exception as e:
                logging.error(f"处理配对失败 {pair_name}: {str(e)}")
                continue
        
        self.results = all_results
        
        # 保存结果
        self.save_results()
        
        logging.info(f"线性模型完成，总计 {len(all_results)} 个模型")
        
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
                'model_name': f"{result['model_type']}_{result['target']}",
                'feature_target_pair': result.get('feature_target_pair', ''),
                'model_type': result['model_type'],
                'target': result['target'],
                'n_features': result['n_features'],
                'r2_score': test_metrics.get('r2_score', 0),
                'rmse': test_metrics.get('rmse', 0),
                'mae': test_metrics.get('mae', 0),
                'mape': test_metrics.get('mape', 0),
                'n_samples': test_metrics.get('n_samples', 0)
            })
        
        # 创建汇总表
        summary_df = self.utils.create_model_summary(summary_data)
        
        # 保存汇总表
        summary_file = os.path.join(self.experiment_dir, 'results', 'linear_models_summary.csv')
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        
        # 保存详细结果
        detailed_file = os.path.join(self.experiment_dir, 'results', 'linear_models_detailed.json')
        import json
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
        
        # 绘制模型比较图
        comparison_plot = os.path.join(self.experiment_dir, 'plots', 'linear_models_comparison.png')
        self.utils.plot_model_comparison(summary_df, 'r2_score', comparison_plot)
        
        logging.info(f"线性模型结果保存完成")
        logging.info(f"汇总文件: {summary_file}")
        logging.info(f"详细结果: {detailed_file}")
        
        # 输出前5个最佳模型
        print("\n=== 线性模型性能排名 (前5名) ===")
        top5 = summary_df.head(5)
        for idx, row in top5.iterrows():
            print(f"{row['rank']}. {row['model_name']}: R² = {row['r2_score']:.4f}, RMSE = {row['rmse']:.6f}")
    
    def analyze_feature_importance(self):
        """分析特征重要性"""
        if not self.results:
            logging.warning("没有模型结果可分析")
            return
        
        # 分析LASSO特征选择结果
        lasso_results = [r for r in self.results if r['model_type'] == 'LASSO']
        
        if lasso_results:
            # 统计特征被选择的频率
            feature_selection_count = {}
            
            for result in lasso_results:
                selected_features = result.get('selected_features', [])
                for feature in selected_features:
                    feature_selection_count[feature] = feature_selection_count.get(feature, 0) + 1
            
            # 创建特征选择频率DataFrame
            if feature_selection_count:
                feature_freq_df = pd.DataFrame([
                    {'feature': feature, 'selection_count': count, 'selection_rate': count/len(lasso_results)}
                    for feature, count in feature_selection_count.items()
                ]).sort_values('selection_count', ascending=False)
                
                # 保存特征选择分析
                feature_analysis_file = os.path.join(self.experiment_dir, 'results', 'feature_selection_analysis.csv')
                feature_freq_df.to_csv(feature_analysis_file, index=False, encoding='utf-8-sig')
                
                # 绘制特征选择频率图
                freq_plot_file = os.path.join(self.experiment_dir, 'plots', 'feature_selection_frequency.png')
                top_features = feature_freq_df.head(20)  # 显示前20个特征
                self.utils.plot_feature_importance(
                    top_features['selection_count'].values,
                    top_features['feature'].values,
                    "LASSO特征选择频率",
                    freq_plot_file
                )
                
                logging.info(f"特征重要性分析完成: {feature_analysis_file}")

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
    linear_models = LinearModels()
    
    # 加载数据
    df = linear_models.utils.load_merged_data(data_file)
    if df is None:
        logging.error("数据加载失败")
        return
    
    # 运行所有线性模型
    results = linear_models.run_all_linear_models(df)
    
    # 分析特征重要性
    linear_models.analyze_feature_importance()
    
    print(f"\n线性模型实验完成，共训练 {len(results)} 个模型")
    print(f"结果保存在: {linear_models.experiment_dir}")

if __name__ == "__main__":
    main()
