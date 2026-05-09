import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ===================== 把你的出租车文件路径填在这里 =====================
DATA_PATH = r"C:\Users\郑xinyi\Downloads\yellow_tripdata_2023-01.parquet"


# ======================================================================

# 读取数据
def load_data():
    df = pd.read_parquet(DATA_PATH)
    return df


# 数据清洗
def clean_data(df):
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["hour"] = df["tpep_pickup_datetime"].dt.hour
    df["weekday"] = df["tpep_pickup_datetime"].dt.weekday
    df = df[(df["fare_amount"] > 0) & (df["trip_distance"] > 0)]
    return df


# -------------------- 图1：小时出行量（M2） --------------------
def plot_hourly_demand(df):
    hourly = df["hour"].value_counts().sort_index()
    plt.figure(figsize=(10, 4))
    sns.lineplot(x=hourly.index, y=hourly.values)
    plt.title("图1：24小时订单量分布")
    plt.savefig("图1_小时订单量.png")
    plt.show()


# -------------------- 图2：车费与距离关系（M2） --------------------
def plot_fare_distance(df):
    plt.figure(figsize=(10, 4))
    sample = df.sample(5000)
    sns.scatterplot(x="trip_distance", y="fare_amount", data=sample, alpha=0.1)
    plt.title("图2：行程距离 vs 车费")
    plt.savefig("图2_车费距离关系.png")
    plt.show()


# -------------------- 图3：热门区域TOP10（M2） --------------------
def plot_top_zones(df):
    top = df["PULocationID"].value_counts().head(10)
    plt.figure(figsize=(10, 4))
    sns.barplot(x=top.index, y=top.values)
    plt.title("图3：上车热门区域TOP10")
    plt.savefig("图3_TOP10区域.png")
    plt.show()


# -------------------- 图4：模型误差（M3） --------------------
def plot_model_error(y_test, y_pred):
    plt.figure(figsize=(10, 4))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.2)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title("图4：预测值 vs 真实值")
    plt.savefig("图4_模型预测效果.png")
    plt.show()


# 训练模型
def train_model(df):
    agg = df.groupby(["PULocationID", "hour"]).size().reset_index(name="demand")
    X = agg[["PULocationID", "hour"]]
    y = agg["demand"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("MAE:", mean_absolute_error(y_test, y_pred))
    return y_test, y_pred


# 主程序
if __name__ == "__main__":
    print("正在读取你的出租车数据...")
    df = load_data()
    df = clean_data(df)

    print("正在生成图1...")
    plot_hourly_demand(df)

    print("正在生成图2...")
    plot_fare_distance(df)

    print("正在生成图3...")
    plot_top_zones(df)

    print("正在训练模型+生成图4...")
    y_test, y_pred = train_model(df)
    plot_model_error(y_test, y_pred)

    print("✅ 全部4张图生成完成！作业达标！")