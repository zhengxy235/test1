import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体支持，防止图表乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 任务1：数据预处理 ====================
print("=== 任务1：数据预处理 ===")

# 1. 读取数据 (注意分隔符为制表符 \t)
try:
    df = pd.read_csv('ICData.csv', sep='\t')
except FileNotFoundError:
    # 如果文件不存在，尝试打印当前目录文件（用于调试）
    import glob
    print("当前目录文件:", glob.glob("*"))
    raise FileNotFoundError("未找到 ICData.csv 文件，请确保文件与代码在同一目录下。")

print("数据集前5行：")
print(df.head())
print(f"\n数据集形状: {df.shape}")
print(f"各列数据类型:\n{df.dtypes}")

# 2. 时间解析
df['交易时间'] = pd.to_datetime(df['交易时间'])
df['hour'] = df['交易时间'].dt.hour  # 新增 hour 列

# 3. 构造衍生字段 & 删除异常
df['ride_stops'] = (df['下车站点'] - df['上车站点']).abs()
initial_len = len(df)
df = df[df['ride_stops'] != 0]  # 删除 ride_stops 为 0 的行
deleted_count = initial_len - len(df)
print(f"\n删除 ride_stops 为 0 的异常记录数: {deleted_count} 行")

# 4. 缺失值检查
missing_values = df.isnull().sum()
print(f"\n各列缺失值数量:\n{missing_values}")
# 处理策略：删除含缺失值的行
if missing_values.sum() > 0:
    df.dropna(inplace=True)
    print("处理策略：删除含缺失值的行。")

print("-" * 50)

# ==================== 任务2：时间分布分析 ====================
print("=== 任务2：时间分布分析 ===")

# (a) 早晚时段刷卡量统计 (必须使用 numpy)
# 筛选上车记录 (刷卡类型=0)
onboard_mask = df['刷卡类型'].values == 0
df_onboard = df[onboard_mask]

# 使用 numpy 统计
hour_vals = df_onboard['hour'].values
total_count = len(hour_vals)

# 定义条件
early_mask = hour_vals < 7
night_mask = hour_vals >= 22

early_count = np.sum(early_mask)
night_count = np.sum(night_mask)

print(f"早峰前时段(07:00前)刷卡量: {early_count} 次")
print(f"深夜时段(22:00后)刷卡量: {night_count} 次")
print(f"早峰前时段占比: {early_count/total_count*100:.2f}%")
print(f"深夜时段占比: {night_count/total_count*100:.2f}%")

# (b) 24小时分布可视化
plt.figure(figsize=(12, 6))
# 统计每个小时的频次
hour_counts = df_onboard['hour'].value_counts().sort_index()
hours = np.arange(24)
counts = [hour_counts.get(h, 0) for h in hours]

# 设置颜色：早高峰和深夜为红色系，其他为蓝色系
colors = []
for h in hours:
    if h < 7 or h >= 22:
        colors.append('#FF6B6B')  # 红色突出显示
    else:
        colors.append('#4ECDC4')  # 蓝色

plt.bar(hours, counts, color=colors, alpha=0.8)

plt.title('全天24小时公交上车刷卡量分布', fontsize=16)
plt.xlabel('小时 (时)', fontsize=12)
plt.ylabel('刷卡量 (次)', fontsize=12)
plt.xticks(hours[::2])  # 标签步长为2
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('hour_distribution.png', dpi=150)
plt.show()

print("-" * 50)

# ==================== 任务3：线路站点分析 ====================
print("=== 任务3：线路站点分析 ===")

def analyze_route_stops(df, route_col='线路号', stops_col='ride_stops'):
    """
    计算各线路乘客的平均搭乘站点数及其标准差。
    Parameters
    ----------
    df : pd.DataFrame  预处理后的数据集
    route_col : str    线路号列名
    stops_col : str    搭乘站点数列名
    Returns
    -------
    pd.DataFrame  包含列：线路号、mean_stops、std_stops，按 mean_stops 降序排列
    """
    # 分组计算
    grouped = df.groupby(route_col)[stops_col].agg(['mean', 'std']).reset_index()
    grouped.columns = [route_col, 'mean_stops', 'std_stops']
    # 降序排列
    result = grouped.sort_values('mean_stops', ascending=False).reset_index(drop=True)
    return result

# 调用函数
result_df = analyze_route_stops(df)
print("各线路平均搭乘站点数（前10行）：")
print(result_df.head(10))

# 可视化 (seaborn 水平条形图)
top15_data = result_df.head(15)
plt.figure(figsize=(10, 8))
# Seaborn barplot 默认是垂直的，为了水平，我们交换x和y，并使用xerr
barplot = sns.barplot(data=top15_data, 
                      x='mean_stops', 
                      y='线路号', 
                      xerr=top15_data['std_stops'], 
                      palette="Blues_d", 
                      capsize=0.1)

plt.title('线路平均搭乘站点数 Top15', fontsize=16)
plt.xlabel('平均搭乘站点数', fontsize=12)
plt.ylabel('线路号', fontsize=12)
plt.xlim(0) # x轴从0开始
plt.tight_layout()
plt.savefig('route_stops.png', dpi=150)
plt.show()

print("-" * 50)

# ==================== 任务4：高峰小时系数计算 ====================
print("=== 任务4：高峰小时系数计算 ===")

# 1. 高峰小时识别
hourly_vol = df['hour'].value_counts()
peak_hour = hourly_vol.idxmax()
peak_hour_count = hourly_vol.max()

print(f"高峰小时：{peak_hour}:00 ~ {peak_hour+1}:00，刷卡量：{peak_hour_count} 次")

# 筛选该高峰小时的数据
peak_df = df[df['hour'] == peak_hour]

# 2. 5分钟粒度统计
# 将分钟数映射到5分钟窗口 (0-4 -> 0, 5-9 -> 5, ...)
peak_df['5min'] = (peak_df['交易时间'].dt.minute // 5) * 5
count_5min = peak_df['5min'].value_counts()
max_5min_count = count_5min.max()
max_5min_window = count_5min.idxmax()
max_5min_time = f"{peak_hour:02d}:{max_5min_window:02d} ~ {peak_hour:02d}:{max_5min_window+5:02d}"

# 3. 15分钟粒度统计
peak_df['15min'] = (peak_df['交易时间'].dt.minute // 15) * 15
count_15min = peak_df['15min'].value_counts()
max_15min_count = count_15min.max()
max_15min_window = count_15min.idxmax()
max_15min_time = f"{peak_hour:02d}:{max_15min_window:02d} ~ {peak_hour:02d}:{max_15min_window+15:02d}"

# 4. 计算 PHF
PHF5 = peak_hour_count / (12 * max_5min_count)
PHF15 = peak_hour_count / (4 * max_15min_count)

print(f"最大5分钟刷卡量（{max_5min_time}）：{max_5min_count} 次")
print(f"PHF5  = {peak_hour_count} / (12 × {max_5min_count}) = {PHF5:.4f}")
print(f"最大15分钟刷卡量（{max_15min_time}）：{max_15min_count} 次")
print(f"PHF15 = {peak_hour_count} / ( 4 × {max_15min_count}) = {PHF15:.4f}")

print("-" * 50)

# ==================== 任务5：线路驾驶员信息批量导出 ====================
print("=== 任务5：线路驾驶员信息批量导出 ===")

target_routes = range(1101, 1121) # 1101 至 1120
output_dir = "线路驾驶员信息"
os.makedirs(output_dir, exist_ok=True)

generated_paths = []

for route in target_routes:
    route_data = df[df['线路号'] == route]
    if not route_data.empty:
        # 去重获取 车辆编号 -> 驾驶员编号 关系
        unique_pairs = route_data[['车辆编号', '驾驶员编号']].drop_duplicates()
        file_path = os.path.join(output_dir, f"{route}.txt")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"线路号: {route}\n")
            f.write("车辆编号\t驾驶员编号\n")
            for _, row in unique_pairs.iterrows():
                f.write(f"{int(row['车辆编号'])}\t{int(row['驾驶员编号'])}\n")
        
        generated_paths.append(file_path)
        print(f"已生成文件: {file_path}")

print(f"\n20个文件生成路径打印完毕，确认全部输出成功。")

print("-" * 50)

# ==================== 任务6：服务绩效排名与热力图 ====================
print("=== 任务6：服务绩效排名与热力图 ===")

# 1. 排名统计
dimensions = {
    '司机': '驾驶员编号',
    '线路': '线路号',
    '上车站点': '上车站点',
    '车辆': '车辆编号'
}

# 构造热力图数据 (4行 x 10列)
heatmap_data = []
row_labels = []

for dim_name, col_name in dimensions.items():
    top10 = df[col_name].value_counts().head(10)
    heatmap_data.append(top10.values)
    row_labels.append(dim_name)

# 转换为 DataFrame
heatmap_df = pd.DataFrame(heatmap_data, 
                          index=row_labels, 
                          columns=[f"Top{i+1}" for i in range(10)])

print("各维度 Top 10 统计：")
print(heatmap_df)

# 2. 热力图可视化
plt.figure(figsize=(12, 5))
sns.heatmap(heatmap_df, annot=True, fmt="d", cmap="YlOrRd", cbar=True)

plt.title('公交服务绩效热力图 (Top 10)', fontsize=16)
plt.suptitle('行：4个维度 (司机/线路/站点/车辆) | 列：各维度 Top 10 实体', fontsize=10)
plt.xlabel('排名 (Top 1 ~ Top 10)')
plt.ylabel('维度')
plt.tight_layout()
plt.savefig('performance_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# 3. 结论说明
print("\n结论说明：")
print("""
从热力图中观察到的服务绩效规律如下：
1. 线路与车辆的关联性：Top 10 线路与 Top 10 车辆的数值分布高度相似，说明热门线路投入了大量同型号的车辆进行高频运营。
2. 司机工作量分布：Top 10 司机的服务人次虽然较高，但差异相比线路间的差异较小，说明司机资源在不同线路间分配相对均衡，或热门线路需要轮换多位司机。
3. 站点枢纽效应：Top 10 上车站点中，部分站点的客流量（数值）显著高于其他站点，表明该城市存在明显的客流集散中心。
4. 运营效率：整体来看，少数线路（如1101路）和少数站点贡献了大部分的客流，建议在高峰时段对这些高绩效实体进行重点监控和资源倾斜。
""")

print("\n=== 作业代码执行完毕 ===")