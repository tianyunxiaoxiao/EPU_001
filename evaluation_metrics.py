import os
import matplotlib.pyplot as plt

def plot_results(results):
    """
    绘制各模型在不同指标下的结果柱状图 (MAE, RMSE, MAPE, Corr, R2)
    并保存到 figures 文件夹
    """
    os.makedirs("figures", exist_ok=True)
    metrics = ["MAE", "RMSE", "MAPE", "Corr", "R2"]

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        plt.bar(results.keys(), [res[metric] for res in results.values()])
        plt.title(f'Model Comparison: {metric}')
        plt.ylabel(metric)
        plt.xlabel('Models')
        fig_path = os.path.join("figures", f"model_comparison_{metric}.png")
        plt.savefig(fig_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"[Info] 图表已保存: {fig_path}")