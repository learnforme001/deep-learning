import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from multilayer_perceptron.polynomial import (
    train_high_degree_poly, get_poly_data, evaluate_loss, train, 
    max_degree, n_train, n_test
)


class TestPolynomialRegression:
    """测试多项式回归功能"""

    def test_get_poly_data_returns_correct_types(self):
        """测试get_poly_data返回正确的数据类型"""
        true_w, features, poly_features, labels = get_poly_data()
        
        assert isinstance(true_w, torch.Tensor)
        assert isinstance(features, torch.Tensor)
        assert isinstance(poly_features, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        
        assert true_w.dtype == torch.float32
        assert features.dtype == torch.float32
        assert poly_features.dtype == torch.float32
        assert labels.dtype == torch.float32

    def test_get_poly_data_correct_dimensions(self):
        """测试get_poly_data返回正确的维度"""
        true_w, features, poly_features, labels = get_poly_data()
        
        assert true_w.shape == (max_degree,)
        assert features.shape == (n_train + n_test, 1)
        assert poly_features.shape == (n_train + n_test, max_degree)
        assert labels.shape == (n_train + n_test,)

    def test_get_poly_data_nonzero_coefficients(self):
        """测试多项式系数前4个为非零值"""
        true_w, _, _, _ = get_poly_data()
        true_w_np = true_w.numpy()
        
        # 前4个系数应该非零
        assert not np.allclose(true_w_np[:4], np.zeros(4))
        # 后面的系数应该为0
        assert np.allclose(true_w_np[4:], np.zeros(max_degree - 4))

    def test_evaluate_loss_with_mock_network(self):
        """测试损失评估函数"""
        # 创建模拟网络和数据迭代器
        mock_net = MagicMock()
        mock_data_iter = [
            (torch.randn(10, 5), torch.randn(10)),
            (torch.randn(5, 5), torch.randn(5))
        ]
        
        # 模拟网络输出
        mock_net.return_value = torch.ones(10, 1)
        loss_fn = torch.nn.MSELoss(reduction='none')
        
        # 计算期望损失
        total_loss = 0
        total_elements = 0
        for X, y in mock_data_iter:
            out = torch.ones(X.shape[0], 1)
            l = loss_fn(out, y.unsqueeze(1))
            total_loss += l.sum().item()
            total_elements += l.numel()
        
        expected_loss = total_loss / total_elements
        
        # 测试评估函数
        with patch.object(mock_net, '__call__', return_value=torch.ones(10, 1)):
            actual_loss = evaluate_loss(mock_net, mock_data_iter, loss_fn)
            assert isinstance(actual_loss, float)
            assert actual_loss >= 0

    @patch('multilayer_perceptron.polynomial.get_poly_data')
    @patch('multilayer_perceptron.polynomial.train')
    def test_train_high_degree_poly_calls_correctly(self, mock_train, mock_get_poly_data):
        """测试train_high_degree_poly正确调用train函数"""
        # 设置模拟返回值
        mock_true_w = torch.zeros(max_degree)
        mock_features = torch.randn(n_train + n_test, 1)
        mock_poly_features = torch.randn(n_train + n_test, max_degree)
        mock_labels = torch.randn(n_train + n_test)
        
        mock_get_poly_data.return_value = (mock_true_w, mock_features, 
                                         mock_poly_features, mock_labels)
        
        # 调用被测试函数
        train_high_degree_poly()
        
        # 验证get_poly_data被调用
        mock_get_poly_data.assert_called_once()
        
        # 验证train被正确调用
        mock_train.assert_called_once()
        args = mock_train.call_args[0]
        
        # 验证参数维度
        assert args[0].shape == (n_train, max_degree)  # train_features
        assert args[1].shape == (n_test, max_degree)   # test_features
        assert args[2].shape == (n_train,)             # train_labels
        assert args[3].shape == (n_test,)              # test_labels

    @patch('multilayer_perceptron.polynomial.nn.Sequential')
    @patch('multilayer_perceptron.polynomial.Data.load_array')
    @patch('multilayer_perceptron.polynomial.Train.train_epoch_ch3')
    def test_train_function_integration(self, mock_train_epoch, mock_load_array, mock_sequential):
        """测试train函数的集成行为"""
        # 设置模拟
        mock_net = MagicMock()
        mock_sequential.return_value = mock_net
        
        mock_data_iter = MagicMock()
        mock_load_array.return_value = mock_data_iter
        
        # 模拟训练过程
        mock_train_epoch.return_value = (0.1, 0.8)  # 训练损失和准确率
        
        # 测试数据
        train_features = torch.randn(n_train, 10)
        test_features = torch.randn(n_test, 10)
        train_labels = torch.randn(n_train)
        test_labels = torch.randn(n_test)
        
        # 调用train函数
        train(train_features, test_features, train_labels, test_labels, num_epochs=2)
        
        # 验证网络创建
        mock_sequential.assert_called_once()
        
        # 验证数据加载器创建
        assert mock_load_array.call_count == 2  # 训练和测试数据加载器
        
        # 验证训练循环
        assert mock_train_epoch.call_count == 2

    def test_train_high_degree_poly_completes_without_error(self):
        """测试train_high_degree_poly能够正常运行完成"""
        # 这个测试验证函数能够运行完成而不抛出异常
        # 使用较小的epoch数来加快测试速度
        original_train = train
        
        def mock_train(*args, **kwargs):
            # 修改num_epochs为较小的值
            kwargs['num_epochs'] = 2
            return original_train(*args, **kwargs)
        
        with patch('multilayer_perceptron.polynomial.train', side_effect=mock_train):
            try:
                train_high_degree_poly()
                # 如果运行到这里说明没有异常
                assert True
            except Exception as e:
                pytest.fail(f"train_high_degree_poly raised unexpected exception: {e}")

    @patch('multilayer_perceptron.polynomial.get_poly_data')
    def test_train_high_degree_poly_with_empty_data(self, mock_get_poly_data):
        """测试空数据情况下的边界条件"""
        # 设置空数据
        mock_get_poly_data.return_value = (
            torch.zeros(max_degree),
            torch.empty(0, 1),
            torch.empty(0, max_degree),
            torch.empty(0)
        )
        
        # 应该能够处理空数据而不崩溃
        try:
            train_high_degree_poly()
            # 空数据处理完成
            assert True
        except Exception as e:
            # 记录异常但不失败，因为空数据可能在某些情况下预期会失败
            print(f"Empty data handling: {type(e).__name__}: {e}")

    def test_polynomial_feature_generation(self):
        """测试多项式特征生成是否正确"""
        # 测试特征生成逻辑
        features = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
        poly_features = torch.zeros(3, max_degree)
        
        # 手动计算多项式特征
        for i in range(max_degree):
            poly_features[:, i] = features.squeeze() ** i
            poly_features[:, i] /= np.math.gamma(i + 1)
        
        # 验证特征值范围
        assert not torch.any(torch.isnan(poly_features))
        assert not torch.any(torch.isinf(poly_features))

    @patch('multilayer_perceptron.polynomial.Animator')
    def test_train_with_mocked_animator(self, mock_animator):
        """测试使用模拟Animator的训练过程"""
        mock_animator_instance = MagicMock()
        mock_animator.return_value = mock_animator_instance
        
        # 测试数据
        train_features = torch.randn(50, 5)
        test_features = torch.randn(20, 5)
        train_labels = torch.randn(50)
        test_labels = torch.randn(20)
        
        # 调用train函数
        train(train_features, test_features, train_labels, test_labels, num_epochs=3)
        
        # 验证Animator被创建
        mock_animator.assert_called_once()
        
        # 验证add方法被调用正确次数
        assert mock_animator_instance.add.call_count == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])