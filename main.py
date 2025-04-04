import os
import pandas as pd
from sklearn.model_selection import train_test_split

from data_processing import generate_processed_epu, load_data
from model_training import train_models
from evaluation_metrics import plot_results

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    # 1. 生成处理后的 EPU 数据
    generate_processed_epu()

    # 2. 加载合并后的数据
    merged_data, _ = load_data()
    merged_data.dropna(subset=["Daily_EPU", "Volatility"], inplace=True)

    if len(merged_data) < 10:
        print("[Error] 数据量不足，无法训练。")
        exit(1)

    # 3. 划分训练集和测试集
    X = merged_data[["Daily_EPU"]]
    y = merged_data["Volatility"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if len(X_test) < 2:
        print("[Error] 测试集过小。")
        exit(1)

    # 4. 模型训练与评估
    results = train_models(X_train, X_test, y_train, y_test)

    # 5. 可视化结果
    plot_results(results)

    # 6. 写入结果文本
    with open("result_001.txt", "w", encoding="utf-8") as f:
        for model_name, metrics in results.items():
            f.write(f"===== {model_name} =====\n")
            f.write(f"MAE: {metrics['MAE']:.3f} | RMSE: {metrics['RMSE']:.3f} | "
                    f"MAPE: {metrics['MAPE']:.3f} | Corr: {metrics['Corr']:.3f} | "
                    f"R2: {metrics['R2']:.3f}\n")
            f.write("Lag Correlations:\n")
            for lag_key, lag_val in metrics["LagCorrs"].items():
                f.write(f"  {lag_key}: {lag_val:.3f}\n")
            f.write("\n")
    print("[Info] 模型训练完成，结果写入 result_001.txt")