# xgboost-stock-predict
基于xgboost的股价上涨预测模型

输入基本面技术指标，预测后15天内是否上涨概率超6%
目前准确率达67%
采用指标：

`python
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
    ]
`

依赖库：
xgboost、sklearn、numpy、pandas
