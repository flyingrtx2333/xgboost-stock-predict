import pandas as pd
import time
from datetime import datetime
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score
from DataSystem import bk


bk_index = 'BK0153'

features = [
        "振幅",
        "涨跌幅",
        'rsi',
        "换手率",
        'kdj_k',
        'kdj_d',
        'kdj_j',
        'wr',
        'macd_line',
        'signal_line'
        # 'macd_histogram'  # 加入此项后准确率反而下降
    ]


def 训练数据生成():
    def future_max(series):
        max_values = []
        n = len(series)
        for i in range(n):
            # 取当前行之后的 15 行
            future_data = series[i + 1:i + 16]
            max_values.append(future_data.max() if not future_data.empty else None)
        return max_values


    start_date_time = '2020-03-22'
    my_dic_result = bk.load_multi_data_from_file(bk_indexs=[bk_index])
    sim_data: dict = bk.produce_simData(my_dic_result, start_date_time=start_date_time)

    rows = []
    for timestamp, stock_data in sim_data.items():
        for gp_code, gp_value in stock_data.items():
            row = {'timestamp': timestamp, 'gp_code': gp_code}
            row.update(gp_value)
            rows.append(row)

    df = pd.DataFrame(rows)
    df['时间'] = df['date_timestamp'].apply(timestamp_to_datetime)
    # 预测未来7天的涨跌幅
    # df['未来7天涨跌幅'] = df.groupby('gp_code')['收盘价'].transform(lambda x: (x.shift(-7) - x) / x)
    # 标注分类：涨（1）或跌（0）
    # df['涨跌方向'] = df['未来7天涨跌幅'].apply(lambda x: 1 if x > 0 else 0)
    # 新增特征：未来15天内是否上涨超过10%
    df['未来15天最高价'] = df.groupby('gp_code')['最高价'].transform(future_max)
    # df['未来15天最低价'] = df.groupby('gp_code')['最低价'].transform(lambda x: x.shift(-1).tailling(window=15).min())

    # 判断未来15天内是否出现上涨超过10%（未来15天最高价比当前收盘价高出10%）
    df['未来15天内是否上涨10%'] = df.apply(lambda row: 1 if row['未来15天最高价'] > row['收盘价'] * 1.1 else 0, axis=1)
    # 判断未来15天内是否出现上涨超过6%（未来15天最高价比当前收盘价高出6%）
    df['未来15天内是否上涨6%'] = df.apply(lambda row: 1 if row['未来15天最高价'] > row['收盘价'] * 1.06 else 0, axis=1)
    # df['未来15天内是否下跌6%'] = df.apply(lambda row: 1 if row['未来15天最低价'] < row['收盘价'] * 0.94 else 0, axis=1)
    print(f"正在保存到本地")
    df.to_csv(f"bk/{bk_index}/train_data.csv", index=False)  # 保存为CSV文件，不包含行索引


def 训练判涨模型():
    df = pd.read_csv(f"bk/{bk_index}/train_data.csv")
    # 划分训练集和测试集
    train_data = df[df['timestamp'] < datetime_to_timestamp('2024-01-01')]  # 设定某个时间点为分界
    # 假设 datetime_to_timestamp 是将日期转换为时间戳的函数
    start_timestamp = datetime_to_timestamp('2023-06-01')
    end_timestamp = datetime_to_timestamp('2024-08-01')

    # 使用括号分隔每个比较条件，并用 & 连接
    test_data = df[(df['timestamp'] >= start_timestamp) & (df['timestamp'] <= end_timestamp)]
    print("划分数据集完毕")

    # 特征和标签

    features_y = '未来15天内是否上涨6%'

    # 准备训练数据和标签
    X_train = []
    y_train = []
    给定过去天数 = 30
    for index in range(给定过去天数, len(train_data)):  # 从第N天开始
        row = []
        for feature in features:
            # 获取过去15天的数据
            past_15_days = train_data[feature].values[index - 给定过去天数:index].tolist()
            row.extend(past_15_days)  # 将过去15天的数据添加到行中
        X_train.append(row)  # 添加完整的15天特征行
        y_train.append(train_data[features_y].iloc[index])  # 目标标签

    X_train = np.array(X_train)  # 转换为 NumPy 数组
    y_train = np.array(y_train)

    # 准备测试数据
    X_test = []
    y_test = []

    for index in range(给定过去天数, len(test_data)):  # 从第N天开始
        row = []
        for feature in features:
            past_15_days = test_data[feature].values[index - 给定过去天数:index].tolist()
            row.extend(past_15_days)
        X_test.append(row)
        y_test.append(test_data[features_y].iloc[index])

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # # 使用单日数据作为训练
    # X_train = train_data[features]
    # y_train = train_data[features_y]
    # X_test = test_data[features]
    # y_test = test_data[features_y]

    # DMatrix 是 XGBoost 的数据格式
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # 模型参数
    params = {
        'objective': 'binary:logistic',
        'max_depth': 8,  # 树的深度
        'eta': 0.03,  # 降低学习率
        'eval_metric': 'auc',
        'subsample': 0.7,  # 随机抽样降低到0.6–0.7之间。增加随机性可以帮助模型提升泛化能力，防止过拟合。
        'colsample_bytree': 0.8,  # 树的特征抽样比例
        'device': 'cuda'  # 启用 GPU 训练
    }

    # 训练模型
    print(f"开始训练")
    model = xgb.train(params, dtrain, num_boost_round=300)
    model.save_model('machine_learning_models/xgboost_up_model.json')  # 可以保存为json或二进制格式

    # 预测
    y_pred = model.predict(dtest)
    y_pred_label = [1 if prob > 0.5 else 0 for prob in y_pred]

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred_label)
    print(f'模型准确率: {accuracy}')


def timestamp_to_datetime(date_timestamp):
    """
    将时间戳转换为时间字符串
    :param date_timestamp:
    :return:
    """
    date = time.strftime("%Y-%m-%d", time.localtime(date_timestamp))
    return date


def datetime_to_timestamp(date_str):
    """
    将"2011-03-22"类似字符串转换为时间戳，用于生成仿真数据时截取开始时间
    :param date_str: "2011-03-22"
    :return: timestamp
    """
    date_object = datetime.strptime(date_str, "%Y-%m-%d")

    # 使用timestamp()函数将datetime对象转换为时间戳
    timestamp = date_object.timestamp()

    return timestamp


if __name__ == '__main__':
    训练判涨模型()
    # 训练判跌模型()
    # 训练数据生成()
