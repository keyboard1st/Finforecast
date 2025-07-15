def generate_rolling_periods(start_month, end_month, predict_window=3, min_train_months=12):
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    def str2date(s):
        return datetime.strptime(s, '%Y%m')
    def date2str(d):
        return d.strftime('%Y%m')

    start = str2date(start_month)
    end = str2date(end_month)

    periods = []
    cur_pred_start = start + relativedelta(months=min_train_months)
    while True:
        pred_start = cur_pred_start
        pred_end = pred_start + relativedelta(months=predict_window-1)
        if pred_end > end:
            break
        train_start = start
        train_end = pred_start - relativedelta(months=1)
        # 训练区间长度检查
        if (train_end - train_start).days < 28:  # 至少1个月
            break
        periods.append((date2str(train_start), date2str(train_end), date2str(pred_start), date2str(pred_end)))
        cur_pred_start = cur_pred_start + relativedelta(months=predict_window)
    return periods

if __name__ == '__main__':
    periods = generate_rolling_periods('201901', '202012', predict_window=3, min_train_months=6)
    for i, (train_start, train_end, pred_start, pred_end) in enumerate(periods):
        print(f"窗口{i+1}: 训练[{train_start}-{train_end}] 预测[{pred_start}-{pred_end}]")