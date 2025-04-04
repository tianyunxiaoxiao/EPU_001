import os

# 数据路径
EPU_PATH = os.getenv("EPU_PATH", ".")
VOL_PATH = os.getenv("VOL_PATH", ".")
PROCESSED_EPU_PATH = "data/processed_EPU.csv"

# 模型参数
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}

OLS_PARAMS = {}

SVR_PARAMS = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale"
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "n_jobs": -1,
    "random_state": 42
}

LSTM_TRAINING_PARAMS = {
    "epochs": 100,  # 降低训练轮数加快实验
    "batch_size": 32
}