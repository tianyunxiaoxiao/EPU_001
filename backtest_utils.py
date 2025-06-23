"""
回测分析工具模块
提供回测分析的通用工具函数
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
import os
import sys
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config

class BacktestUtils:
    """回测工具类"""
    
    def __init__(self):
        self.setup_chinese_font()
    
    def load_stock_data(self, stock_data_file=None):
        """
        加载股指价格数据
        
        Args:
            stock_data_file: 股指数据文件路径，如果为None则使用默认路径
            
        Returns:
            pd.DataFrame: 股指价格数据
        """
        try:
            if stock_data_file is None:
                stock_data_file = os.path.join(Config.RAW_NEWS_DIR, "stock_data.csv")
            
            if not os.path.exists(stock_data_file):
                logging.warning(f"股指数据文件不存在: {stock_data_file}")
                return None
            
            stock_data = pd.read_csv(stock_data_file)
            
            # 确保Date列是datetime类型
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            
            # 按日期排序
            stock_data = stock_data.sort_values('Date')
            
            logging.info(f"成功加载股指数据: {len(stock_data)} 条记录")
            return stock_data
            
        except Exception as e:
            logging.error(f"加载股指数据失败: {str(e)}")
            return None
    
    def prepare_backtest_data(self, epu_data, stock_data, price_column='Close'):
        """
        准备回测数据，合并EPU数据和价格数据
        
        Args:
            epu_data: EPU数据DataFrame
            stock_data: 股指数据DataFrame
            price_column: 价格列名
            
        Returns:
            pd.DataFrame: 合并后的回测数据
        """
        try:
            if epu_data is None or stock_data is None:
                logging.error("EPU数据或股指数据为空")
                return None
            
            # 确保日期列格式一致
            if 'date' in epu_data.columns:
                epu_data['Date'] = pd.to_datetime(epu_data['date'])
            elif 'Date' not in epu_data.columns:
                logging.error("EPU数据中找不到日期列")
                return None
            
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            
            # 按日期合并数据
            merged_data = pd.merge(
                epu_data, 
                stock_data[['Date', price_column, 'returns']], 
                on='Date', 
                how='inner'
            )
            
            if len(merged_data) == 0:
                logging.error("合并后的数据为空，请检查日期范围是否匹配")
                return None
            
            # 确保数据按日期排序
            merged_data = merged_data.sort_values('Date')
            
            logging.info(f"成功准备回测数据: {len(merged_data)} 条记录")
            return merged_data
            
        except Exception as e:
            logging.error(f"准备回测数据失败: {str(e)}")
            return None
        
    def setup_chinese_font(self):
        """设置中文字体"""
        try:
            font_path = Config.FONT_PATH
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            logging.warning(f"中文字体设置失败: {str(e)}")
    
    def calculate_returns(self, prices):
        """计算收益率"""
        if len(prices) < 2:
            return pd.Series(dtype=float)
        
        prices = pd.Series(prices) if not isinstance(prices, pd.Series) else prices
        returns = prices.pct_change().dropna()
        return returns
    
    def calculate_log_returns(self, prices):
        """计算对数收益率"""
        if len(prices) < 2:
            return pd.Series(dtype=float)
        
        prices = pd.Series(prices) if not isinstance(prices, pd.Series) else prices
        log_returns = np.log(prices / prices.shift(1)).dropna()
        return log_returns
    
    def calculate_cumulative_returns(self, returns):
        """计算累计收益率"""
        returns = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
        cum_returns = (1 + returns).cumprod() - 1
        return cum_returns
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.0):
        """计算夏普比率"""
        returns = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
        
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # 假设年化无风险收益率
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)
        return sharpe_ratio
    
    def calculate_sortino_ratio(self, returns, risk_free_rate=0.0):
        """计算索提诺比率"""
        returns = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
        
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return np.inf if excess_returns.mean() > 0 else 0.0
        
        sortino_ratio = excess_returns.mean() / downside_returns.std() * np.sqrt(252)
        return sortino_ratio
    
    def calculate_max_drawdown(self, returns):
        """计算最大回撤"""
        returns = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
        
        if len(returns) == 0:
            return 0.0
        
        cum_returns = self.calculate_cumulative_returns(returns)
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / (1 + running_max)
        max_drawdown = drawdown.min()
        
        return max_drawdown
    
    def calculate_calmar_ratio(self, returns):
        """计算卡玛比率"""
        returns = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
        
        if len(returns) == 0:
            return 0.0
        
        annual_return = (1 + returns.mean()) ** 252 - 1
        max_dd = abs(self.calculate_max_drawdown(returns))
        
        if max_dd == 0:
            return np.inf if annual_return > 0 else 0.0
        
        calmar_ratio = annual_return / max_dd
        return calmar_ratio
    
    def calculate_volatility(self, returns, annualized=True):
        """计算波动率"""
        returns = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
        
        if len(returns) == 0:
            return 0.0
        
        volatility = returns.std()
        if annualized:
            volatility *= np.sqrt(252)
        
        return volatility
    
    def calculate_var(self, returns, confidence_level=0.05):
        """计算风险价值(VaR)"""
        returns = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
        
        if len(returns) == 0:
            return 0.0
        
        var = returns.quantile(confidence_level)
        return var
    
    def calculate_cvar(self, returns, confidence_level=0.05):
        """计算条件风险价值(CVaR)"""
        returns = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
        
        if len(returns) == 0:
            return 0.0
        
        var = self.calculate_var(returns, confidence_level)
        cvar = returns[returns <= var].mean()
        return cvar
    
    def calculate_hit_rate(self, actual_returns, predicted_direction):
        """计算方向预测准确率"""
        actual_returns = pd.Series(actual_returns) if not isinstance(actual_returns, pd.Series) else actual_returns
        predicted_direction = pd.Series(predicted_direction) if not isinstance(predicted_direction, pd.Series) else predicted_direction
        
        if len(actual_returns) != len(predicted_direction) or len(actual_returns) == 0:
            return 0.0
        
        actual_direction = np.sign(actual_returns)
        correct_predictions = (actual_direction == predicted_direction).sum()
        hit_rate = correct_predictions / len(actual_returns)
        
        return hit_rate
    
    def create_performance_metrics(self, returns, benchmark_returns=None):
        """创建性能指标汇总"""
        returns = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
        
        if len(returns) == 0:
            return {}
        
        metrics = {
            '总收益率': self.calculate_cumulative_returns(returns).iloc[-1],
            '年化收益率': (1 + returns.mean()) ** 252 - 1,
            '年化波动率': self.calculate_volatility(returns),
            '夏普比率': self.calculate_sharpe_ratio(returns),
            '索提诺比率': self.calculate_sortino_ratio(returns),
            '最大回撤': self.calculate_max_drawdown(returns),
            '卡玛比率': self.calculate_calmar_ratio(returns),
            'VaR(5%)': self.calculate_var(returns),
            'CVaR(5%)': self.calculate_cvar(returns),
            '胜率': (returns > 0).mean(),
            '交易次数': len(returns)
        }
        
        # 如果有基准，计算相对指标
        if benchmark_returns is not None:
            benchmark_returns = pd.Series(benchmark_returns) if not isinstance(benchmark_returns, pd.Series) else benchmark_returns
            
            if len(benchmark_returns) == len(returns):
                excess_returns = returns - benchmark_returns
                # 先计算Beta
                beta = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns) if np.var(benchmark_returns) > 0 else 0
                benchmark_annual_return = (1 + benchmark_returns.mean()) ** 252 - 1
                alpha = metrics['年化收益率'] - beta * benchmark_annual_return
                
                metrics.update({
                    '基准收益率': self.calculate_cumulative_returns(benchmark_returns).iloc[-1],
                    '超额收益率': self.calculate_cumulative_returns(excess_returns).iloc[-1],
                    '信息比率': excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0,
                    'Beta': beta,
                    'Alpha': alpha
                })
        
        return metrics
    
    def plot_performance(self, returns, benchmark_returns=None, title="策略表现", save_path=None):
        """绘制策略表现图"""
        returns = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
        
        if len(returns) == 0:
            logging.warning("无收益率数据可绘制")
            return None
        
        # 计算累计收益率
        cum_returns = self.calculate_cumulative_returns(returns)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 累计收益率曲线
        ax1.plot(cum_returns.index, cum_returns.values, label='策略', linewidth=2)
        if benchmark_returns is not None:
            benchmark_returns = pd.Series(benchmark_returns) if not isinstance(benchmark_returns, pd.Series) else benchmark_returns
            if len(benchmark_returns) == len(returns):
                bench_cum_returns = self.calculate_cumulative_returns(benchmark_returns)
                ax1.plot(bench_cum_returns.index, bench_cum_returns.values, label='基准', linewidth=2)
        
        ax1.set_title('累计收益率')
        ax1.set_ylabel('累计收益率')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 回撤曲线
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / (1 + running_max)
        ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax2.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        ax2.set_title('回撤曲线')
        ax2.set_ylabel('回撤')
        ax2.grid(True, alpha=0.3)
        
        # 收益率分布
        ax3.hist(returns.values, bins=50, alpha=0.7, density=True)
        ax3.axvline(returns.mean(), color='red', linestyle='--', label=f'均值: {returns.mean():.4f}')
        ax3.axvline(returns.median(), color='green', linestyle='--', label=f'中位数: {returns.median():.4f}')
        ax3.set_title('收益率分布')
        ax3.set_xlabel('日收益率')
        ax3.set_ylabel('密度')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 滚动夏普比率
        rolling_sharpe = returns.rolling(window=60).apply(
            lambda x: self.calculate_sharpe_ratio(x) if len(x) == 60 else np.nan
        )
        ax4.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title('60日滚动夏普比率')
        ax4.set_ylabel('夏普比率')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"策略表现图保存至: {save_path}")
        
        return fig
    
    def plot_monthly_returns(self, returns, title="月度收益率热力图", save_path=None):
        """绘制月度收益率热力图"""
        returns = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
        
        if len(returns) == 0:
            logging.warning("无收益率数据可绘制")
            return None
        
        # 确保索引是日期格式
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)
        
        # 按月分组计算收益率
        monthly_returns = returns.groupby([returns.index.year, returns.index.month]).apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # 重新构造索引
        monthly_returns.index = pd.MultiIndex.from_tuples(monthly_returns.index, names=['Year', 'Month'])
        
        # 创建数据透视表
        monthly_table = monthly_returns.unstack(level='Month')
        monthly_table.columns = ['1月', '2月', '3月', '4月', '5月', '6月',
                                '7月', '8月', '9月', '10月', '11月', '12月']
        
        # 绘制热力图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 使用红绿色调
        import matplotlib.colors as mcolors
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "rg", ["red", "white", "green"], N=256
        )
        
        im = ax.imshow(monthly_table.values, cmap=cmap, aspect='auto')
        
        # 设置坐标轴
        ax.set_xticks(range(len(monthly_table.columns)))
        ax.set_xticklabels(monthly_table.columns)
        ax.set_yticks(range(len(monthly_table.index)))
        ax.set_yticklabels(monthly_table.index)
        
        # 添加数值标签
        for i in range(len(monthly_table.index)):
            for j in range(len(monthly_table.columns)):
                value = monthly_table.iloc[i, j]
                if not pd.isna(value):
                    text_color = 'white' if abs(value) > 0.05 else 'black'
                    ax.text(j, i, f'{value:.2%}', ha="center", va="center", color=text_color)
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, label='月度收益率')
        
        ax.set_title(title)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"月度收益率热力图保存至: {save_path}")
        
        return fig
    
    def create_tear_sheet(self, returns, benchmark_returns=None, strategy_name="策略", save_dir=None):
        """创建策略分析报告"""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 性能指标
        metrics = self.create_performance_metrics(returns, benchmark_returns)
        
        # 保存指标到CSV
        if save_dir:
            metrics_df = pd.DataFrame(list(metrics.items()), columns=['指标', '数值'])
            metrics_file = os.path.join(save_dir, f'{strategy_name}_metrics.csv')
            metrics_df.to_csv(metrics_file, index=False, encoding='utf-8-sig')
            logging.info(f"性能指标保存至: {metrics_file}")
        
        # 绘制主要表现图
        if save_dir:
            performance_plot = os.path.join(save_dir, f'{strategy_name}_performance.png')
            self.plot_performance(returns, benchmark_returns, f"{strategy_name}表现", performance_plot)
            
            # 月度收益率热力图
            monthly_plot = os.path.join(save_dir, f'{strategy_name}_monthly_returns.png')
            self.plot_monthly_returns(returns, f"{strategy_name}月度收益率", monthly_plot)
        
        return metrics
    
    def compare_strategies(self, strategy_returns_dict, benchmark_returns=None, save_path=None):
        """比较多个策略"""
        if not strategy_returns_dict:
            logging.warning("没有策略数据可比较")
            return None
        
        # 计算各策略指标
        comparison_data = []
        for strategy_name, returns in strategy_returns_dict.items():
            metrics = self.create_performance_metrics(returns, benchmark_returns)
            metrics['策略名称'] = strategy_name
            comparison_data.append(metrics)
        
        # 创建比较表
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.set_index('策略名称')
        
        # 保存比较结果
        if save_path:
            comparison_file = save_path.replace('.png', '.csv')
            comparison_df.to_csv(comparison_file, encoding='utf-8-sig')
            logging.info(f"策略比较结果保存至: {comparison_file}")
        
        # 绘制比较图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 累计收益率比较
        for strategy_name, returns in strategy_returns_dict.items():
            cum_returns = self.calculate_cumulative_returns(returns)
            ax1.plot(cum_returns.index, cum_returns.values, label=strategy_name, linewidth=2)
        
        if benchmark_returns is not None:
            bench_cum_returns = self.calculate_cumulative_returns(benchmark_returns)
            ax1.plot(bench_cum_returns.index, bench_cum_returns.values, label='基准', linewidth=2, linestyle='--')
        
        ax1.set_title('累计收益率比较')
        ax1.set_ylabel('累计收益率')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 夏普比率比较
        sharpe_ratios = [self.calculate_sharpe_ratio(returns) for returns in strategy_returns_dict.values()]
        ax2.bar(strategy_returns_dict.keys(), sharpe_ratios)
        ax2.set_title('夏普比率比较')
        ax2.set_ylabel('夏普比率')
        ax2.grid(True, alpha=0.3)
        
        # 最大回撤比较
        max_drawdowns = [abs(self.calculate_max_drawdown(returns)) for returns in strategy_returns_dict.values()]
        ax3.bar(strategy_returns_dict.keys(), max_drawdowns, color='red', alpha=0.7)
        ax3.set_title('最大回撤比较')
        ax3.set_ylabel('最大回撤')
        ax3.grid(True, alpha=0.3)
        
        # 年化收益率vs波动率散点图
        annual_returns = [(1 + pd.Series(returns).mean()) ** 252 - 1 for returns in strategy_returns_dict.values()]
        volatilities = [self.calculate_volatility(returns) for returns in strategy_returns_dict.values()]
        
        ax4.scatter(volatilities, annual_returns, s=100)
        for i, strategy_name in enumerate(strategy_returns_dict.keys()):
            ax4.annotate(strategy_name, (volatilities[i], annual_returns[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax4.set_xlabel('年化波动率')
        ax4.set_ylabel('年化收益率')
        ax4.set_title('收益率-风险散点图')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('策略比较分析', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"策略比较图保存至: {save_path}")
        
        return comparison_df

def main():
    """测试函数"""
    # 创建模拟数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    returns = np.random.normal(0.001, 0.02, len(dates))
    
    utils = BacktestUtils()
    
    # 测试性能指标计算
    metrics = utils.create_performance_metrics(returns)
    print("性能指标:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # 测试绘图功能
    test_returns = pd.Series(returns, index=dates)
    utils.plot_performance(test_returns, title="测试策略表现")
    plt.show()

if __name__ == "__main__":
    main()
