郑新译 - 25361051 - 第三次人工智能编程作业
1. 任务拆解与 AI 协作策略
我将6 项任务按 “数据预处理→统计计算→可视化→文件导出” 的顺序分步拆解，没有一次性让 AI 生成全部代码。
先让 AI 完成任务 1 数据预处理（读取、时间转换、异常值删除、缺失值处理），确保数据基础正确；
再依次完成任务 2 时间分析、任务 3 线路分析、任务 4 高峰小时系数；
然后完成任务 5 文件导出、任务 6 热力图；
每一步都明确约束：必须用指定库、必须按函数签名、必须输出指定格式。
2. 核心 Prompt 迭代记录
初代 Prompt
帮我写公交 IC 卡数据分析代码，完成 6 个任务，画图并导出文件。
AI 生成的问题
任务 2 时段统计只用了 pandas，没有用 numpy；
任务 3 函数签名与题目不一致；
任务 4 没有自动找高峰小时，直接固定时段；
任务 2 柱状图用了 seaborn，不符合要求。
优化后的 Prompt
严格按第三次作业要求写代码：任务 2 必须用 numpy 做时段统计；任务 3 函数签名完全按题目不能改；任务 4 先自动找高峰小时再算 PHF；任务 2 必须用 matplotlib 画柱状图；代码加逐行中文注释，输出指定图片和文件夹。
3. Debug 记录
报错现象
图表中文乱码显示方框，任务 5 导出 txt 提示路径不存在。
解决过程
中文乱码：在代码开头添加 matplotlib 中文字体设置；
文件路径错误：先判断并创建 “线路驾驶员信息” 文件夹，再写入文件；
重新运行后图表正常、文件成功导出。
4. 人工代码审查
# 定义高峰小时系数计算函数
def phf_analysis(df):
    # 提取交易时间中的小时信息
    df['小时'] = df['交易时间'].dt.hour
    # 找出刷卡量最多的小时作为高峰小时
    peak_hour = df['小时'].value_counts().idxmax()
    # 筛选出高峰小时内的所有数据
    peak_df = df[df['小时'] == peak_hour]
    # 统计高峰小时总刷卡次数
    peak_count = len(peak_df)

    # 按5分钟为一个窗口进行分组
    peak_df['5分钟'] = peak_df['交易时间'].dt.minute // 5
    # 获取5分钟窗口内的最大刷卡量
    max_5min = peak_df['5分钟'].value_counts().max()
    # 按公式计算PHF5
    phf5 = peak_count / (12 * max_5min)

    # 按15分钟为一个窗口进行分组
    peak_df['15分钟'] = peak_df['交易时间'].dt.minute // 15
    # 获取15分钟窗口内的最大刷卡量
    max_15min = peak_df['15分钟'].value_counts().max()
    # 按公式计算PHF15
    phf15 = peak_count / (4 * max_15min)

    # 按要求格式输出结果
    print(f"高峰小时：{peak_hour}:00 ~ {peak_hour+1}:00，刷卡量：{peak_count} 次")
    print(f"最大5分钟刷卡量：{max_5min} 次")
    print(f"PHF5 = {peak_count}/(12×{max_5min}) = {phf5:.4f}")
    print(f"最大15分钟刷卡量：{max_15min} 次")
    print(f"PHF15 = {peak_count}/(4×{max_15min}) = {phf15:.4f}")
    return phf5, phf15
