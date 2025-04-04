import os
import pandas as pd
from config import EPU_PATH, VOL_PATH, PROCESSED_EPU_PATH

def generate_processed_epu():
    os.makedirs("data", exist_ok=True)

    # 直接读取all_epu_data_001.xlsx文件
    epu_file = "all_epu_data_001.xlsx"
    if not os.path.exists(epu_file):
        raise FileNotFoundError(f"未找到 {epu_file} 文件!")

    # 读取EPU数据
    epu_data = pd.read_excel(epu_file)
    
    # 确保数据含有必要的列
    if "日期" not in epu_data.columns or "EPU" not in epu_data.columns:
        raise ValueError(f"{epu_file} 中缺少必要的列 (日期或EPU)")
    
    # 重命名列以匹配后续处理
    epu_data.rename(columns={"日期": "Date", "EPU": "Daily_EPU"}, inplace=True)
    
    # 转换日期格式
    epu_data["Date"] = pd.to_datetime(epu_data["Date"], errors="coerce")
    epu_data.dropna(subset=["Date", "Daily_EPU"], inplace=True)

    # 保存处理后的数据
    out_df = epu_data[["Date", "Daily_EPU"]].copy()
    out_df.to_csv(PROCESSED_EPU_PATH, index=False)
    print(f"[Info] EPU 数据已保存到 {PROCESSED_EPU_PATH}")

def load_data():
    if not os.path.exists(PROCESSED_EPU_PATH):
        raise FileNotFoundError(f"{PROCESSED_EPU_PATH} 不存在, 请先执行 generate_processed_epu()")

    epu_df = pd.read_csv(PROCESSED_EPU_PATH)
    epu_df["Date"] = pd.to_datetime(epu_df["Date"], errors="coerce")
    epu_df.dropna(subset=["Date", "Daily_EPU"], inplace=True)

    vol_files = [f for f in os.listdir(VOL_PATH) if f.endswith('.csv') and 'volatility' in f]
    if not vol_files:
        raise FileNotFoundError(f"在 {VOL_PATH} 未找到包含 'volatility' 的文件!")

    vol_list = []
    for file in vol_files:
        df = pd.read_csv(os.path.join(VOL_PATH, file))
        df = df.loc[:, ~df.columns.duplicated()]
        vol_list.append(df)

    vol_data = pd.concat(vol_list, ignore_index=True)
    vol_data.rename(columns={"DateTime": "Date"}, inplace=True)
    vol_data["Date"] = pd.to_datetime(vol_data["Date"], errors="coerce")
    vol_data.dropna(subset=["Date"], inplace=True)

    merged_data = pd.merge(epu_df, vol_data, on="Date", how="inner")
    merged_data.dropna(subset=["Daily_EPU", "Volatility"], inplace=True)
    monthly_data = merged_data.resample("M", on="Date").mean().reset_index()

    return merged_data, monthly_data