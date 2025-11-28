import pandas as pd
import numpy as np
import torch
from torch import nn
from basic import DownloadHelper, Data
from matplotlib import pyplot as plt

def kaggle_house_price():
    """Kaggle房价预测数据集的加载与预处理"""
    DATA_HUB = DownloadHelper.DATA_HUB
    DATA_HUB['kaggle_house_train'] = (DownloadHelper.DATA_URL + 'kaggle_house_pred_train.csv', '585e9cc93e70b39160e7921475f9bcd7d31219ce')
    DATA_HUB['kaggle_house_test'] = (DownloadHelper.DATA_URL + 'kaggle_house_pred_test.csv', 'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

    train_data = pd.read_csv(DownloadHelper.download('kaggle_house_train'))
    test_data = pd.read_csv(DownloadHelper.download('kaggle_house_test'))

    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

    # 数据预处理——标准化
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std())
    )
    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    # 将离散值转换为独热编码
    all_features = pd.get_dummies(all_features, dummy_na=True)
    # 确保所有特征都是数值类型
    all_features = all_features.astype('float32')
    
    n_train = train_data.shape[0]
    train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
    test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
    train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)
    loss = nn.MSELoss()
    in_features = train_features.shape[1]

    def get_net():
        net = nn.Sequential(nn.Linear(in_features,1))
        return net

    def log_rmse(net, features, labels):
        clipped_preds = torch.clamp(net(features), 1, float('inf'))
        rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
        return rmse.item()
    
    def train(net, train_features, train_labels, test_features, test_labels,
            num_epochs, learning_rate, weight_decay, batch_size):
        train_ls, test_ls = [], []
        train_iter = Data.load_array((train_features, train_labels), batch_size)
        # 这里使用的是Adam优化算法
        optimizer = torch.optim.Adam(net.parameters(),
                                    lr = learning_rate,
                                    weight_decay = weight_decay)
        for epoch in range(num_epochs):
            for X, y in train_iter:
                optimizer.zero_grad()
                l = loss(net(X), y)
                l.backward()
                optimizer.step()
            train_ls.append(log_rmse(net, train_features, train_labels))
            if test_labels is not None:
                test_ls.append(log_rmse(net, test_features, test_labels))
        return train_ls, test_ls
        
    def get_k_fold_data(k, i, X ,y):
        assert k>1
        fold_size = X.shape[0] // k
        X_train, y_train = None, None
        for j in range(k):
            idx = slice(j * fold_size, (j + 1) * fold_size)
            X_part, y_part = X[idx, :], y[idx]
            if j == i:
                X_valid, y_valid = X_part, y_part
            elif X_train is None:
                X_train, y_train = X_part, y_part
            else:
                X_train = torch.cat([X_train, X_part], 0)
                y_train = torch.cat([y_train, y_part], 0)
        return X_train, y_train, X_valid, y_valid
    
    def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
        train_l_sum, valid_l_sum = 0, 0
        for i in range(k):
            data = get_k_fold_data(k, i, X_train, y_train)
            net = get_net()
            train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                    weight_decay, batch_size)
            train_l_sum += train_ls[-1]
            valid_l_sum += valid_ls[-1]
            if i == 0:
                import os
                os.makedirs('outputs', exist_ok=True)
                epochs = list(range(1, num_epochs + 1))
                plt.plot(epochs, train_ls, label='train')
                plt.plot(epochs, valid_ls, label='valid')
                plt.xlabel('epoch')
                plt.ylabel('rmse')
                plt.xlim([1, num_epochs])
                plt.yscale('log')
                plt.legend()
                plt.savefig('outputs/kaggle_k_fold.png', dpi=200, bbox_inches='tight')
                plt.close()
                print('图片已保存到: outputs/kaggle_k_fold.png')
            print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
                f'验证log rmse{float(valid_ls[-1]):f}')
        return train_l_sum / k, valid_l_sum / k

    k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
    # train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
    #                         weight_decay, batch_size)
    # print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
    #     f'平均验证log rmse: {float(valid_l):f}')
    
    def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
        import os
        os.makedirs('outputs', exist_ok=True)
        net = get_net()
        train_ls, _ = train(net, train_features, train_labels, None, None,
                            num_epochs, lr, weight_decay, batch_size)
        epochs = np.arange(1, num_epochs + 1)
        plt.plot(epochs, train_ls, label='train')
        plt.xlabel('epoch')
        plt.ylabel('log rmse')
        plt.xlim([1, num_epochs])
        plt.yscale('log')
        plt.legend()
        plt.savefig('outputs/kaggle_train_pred.png', dpi=200, bbox_inches='tight')
        plt.close()
        print('图片已保存到: outputs/kaggle_train_pred.png')
        print(f'训练log rmse：{float(train_ls[-1]):f}')
        # 将网络应用于测试集。
        preds = net(test_features).detach().numpy()
        # 将其重新格式化以导出到Kaggle
        test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
        submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
        submission.to_csv('submission.csv', index=False)
    
    train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)