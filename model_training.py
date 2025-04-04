import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from tensorflow import keras
from keras import layers
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)
from config import (
    XGBOOST_PARAMS, OLS_PARAMS, SVR_PARAMS, RF_PARAMS,
    LSTM_TRAINING_PARAMS
)

def train_models(X_train, X_test, y_train, y_test):
    models = {
        "XGBoost": XGBRegressor(**XGBOOST_PARAMS),
        "OLS": LinearRegression(**OLS_PARAMS),
        "SVR": SVR(**SVR_PARAMS),
        "RF": RandomForestRegressor(**RF_PARAMS),
        "LSTM": keras.Sequential([
            keras.Input(shape=(X_train.shape[1], 1)),
            layers.LSTM(64, return_sequences=True),
            layers.LSTM(32),
            layers.Dense(1)
        ])
    }

    results = {}

    for name, model in models.items():
        if name == "LSTM":
            X_train_lstm = np.expand_dims(X_train.values, axis=2)
            X_test_lstm = np.expand_dims(X_test.values, axis=2)

            model.compile(optimizer="adam", loss="mse")
            model.fit(
                X_train_lstm, y_train.values,
                epochs=LSTM_TRAINING_PARAMS["epochs"],
                batch_size=LSTM_TRAINING_PARAMS["batch_size"],
                verbose=0
            )
            predictions = model.predict(X_test_lstm).flatten()
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

        results[name] = compute_metrics(y_test, predictions)

    return results

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0

    lag_corrs = {}
    for lag in range(1, 7):
        shifted_y = y_true.shift(lag)
        valid = ~shifted_y.isna()
        if valid.sum() > 2:
            lag_corrs[f"lag_{lag}"] = np.corrcoef(shifted_y[valid], y_pred[valid])[0, 1]
        else:
            lag_corrs[f"lag_{lag}"] = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Corr": corr,
        "R2": r2,
        "LagCorrs": lag_corrs
    }