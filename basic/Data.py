from torch.utils import data

class Data:
    @classmethod
    def load_array(cls, data_arrays, batch_size, is_train=True):
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)