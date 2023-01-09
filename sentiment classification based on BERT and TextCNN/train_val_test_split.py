from sklearn.model_selection import train_test_split
import pandas as pd


def train_test_val_split(data, ratio_train, ratio_test, ratio_val):
    # 欠采样
    data_pos = data[data[0] == 1]
    data_neg = data[data[0] == 0]
    data_neg = data_neg.sample(frac=0.5, random_state=111, axis=0)
    data = pd.concat([data_pos, data_neg], axis=0)
    data = data.reset_index(drop=True)  # 重设索引

    print(data[1].map(len).max())

    train, middle = train_test_split(data, train_size=ratio_train, test_size=ratio_test + ratio_val)
    ratio = ratio_val / (1 - ratio_train)
    test, validation = train_test_split(middle, test_size=ratio)
    return train, test, validation


if __name__ == '__main__':
    data_path = 'dzdp_data/original.csv'
    out_train_path = 'dzdp_data/train.csv'
    out_test_path = 'dzdp_data/test.csv'
    out_val_path = 'dzdp_data/val.csv'

    data_df = pd.read_csv(data_path, encoding='ANSI', header=None)

    train, test, validation = train_test_val_split(data_df, ratio_train=0.6, ratio_test=0.2, ratio_val=0.2)
    train.to_csv(out_train_path, sep=',', index=False, header=True)
    test.to_csv(out_test_path, sep=',', index=False, header=True)
    validation.to_csv(out_val_path, sep=',', index=False, header=True)
